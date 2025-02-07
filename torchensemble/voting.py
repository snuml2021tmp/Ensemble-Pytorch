"""
  In voting-based ensemble, each base estimator is trained independently,
  and the final prediction takes the average over predictions from all base
  estimators.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings
from joblib import Parallel, delayed

from ._base import BaseClassifier, BaseRegressor
from ._base import torchensemble_model_doc
from .utils import io
from .utils import set_module
from .utils import operator as op
import numpy as np

__all__ = ["VotingClassifier", "VotingRegressor"]


def _parallel_fit_per_epoch(
    train_loader,
    estimator,
    cur_lr,
    optimizer,
    criterion,
    idx,
    epoch,
    log_interval,
    device,
    is_classification,
):
    """
    Private function used to fit base estimators in parallel.

    WARNING: Parallelization when fitting large base estimators may cause
    out-of-memory error.
    """

    if cur_lr:
        # Parallelization corrupts the binding between optimizer and scheduler
        set_module.update_lr(optimizer, cur_lr)

    for batch_idx, (data, target) in enumerate(train_loader):

        batch_size = data.size(0)
        data, target = data.to(device), target.to(device)
        # added for ML2021
        data = (data, target)

        optimizer.zero_grad()
        output = estimator(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Print training status
        if batch_idx % log_interval == 0:

            # Classification
            if is_classification:
                _, predicted = torch.max(output.data, 1)
                correct = (predicted == target).sum().item()

                msg = (
                    "Estimator: {:03d} | Epoch: {:03d} | Batch: {:03d}"
                    " | LR: {:.5f} | Loss: {:.5f} | Correct: {:d}/{:d}"
                )
                print(
                    msg.format(
                        idx, epoch, batch_idx, cur_lr, loss, correct, batch_size
                    )
                )
            # Regression
            else:
                msg = (
                    "Estimator: {:03d} | Epoch: {:03d} | Batch: {:03d}"
                    " | Loss: {:.5f}"
                )
                print(msg.format(idx, epoch, batch_idx, loss))

    return estimator, optimizer


@torchensemble_model_doc(
    """Implementation on the VotingClassifier.""", "model"
)
class VotingClassifier(BaseClassifier):
    @torchensemble_model_doc(
        """Implementation on the data forwarding in VotingClassifier.""",
        "classifier_forward",
    )
    def forward(self, x):
        # Average over class distributions from all base estimators.
        outputs = [
            F.softmax(estimator(x), dim=1) for estimator in self.estimators_
        ]
        proba = op.average(outputs)

        return proba

    @torchensemble_model_doc(
        """Set the attributes on optimizer for VotingClassifier.""",
        "set_optimizer",
    )
    def set_optimizer(self, optimizer_name, **kwargs):
        super().set_optimizer(optimizer_name, **kwargs)

    @torchensemble_model_doc(
        """Set the attributes on scheduler for VotingClassifier.""",
        "set_scheduler",
    )
    def set_scheduler(self, scheduler_name, **kwargs):
        super().set_scheduler(scheduler_name, **kwargs)

    @torchensemble_model_doc(
        """Implementation on the training stage of VotingClassifier.""", "fit"
    )
    def fit(
        self,
        train_loader,
        epochs=100,
        log_interval=100,
        test_loader=None,
        save_model=True,
        save_dir=None,
        retrain=False,
        loaded_optimizers=None,
        loaded_scheduler=None,
        loaded_epoch=0,
        loaded_est_idx=-1,
        loaded_best_acc=-1,
    ):

        self._validate_parameters(epochs, log_interval)
        self.n_outputs = self._decide_n_outputs(train_loader)

        # Instantiate a pool of base estimators, optimizers, and schedulers.
        if not retrain:
            estimators = []
            for _ in range(self.n_estimators):
                estimators.append(self._make_estimator())
        else:
            estimators = self.estimators_

        if not retrain:
            optimizers = []
            for i in range(self.n_estimators):
                optimizers.append(
                    set_module.set_optimizer(
                        estimators[i], self.optimizer_name, **self.optimizer_args
                    )
                )
        else:
            assert loaded_optimizers is not None
            optimizers = loaded_optimizers

        if self.use_scheduler_:
            if not retrain:
                scheduler_ = set_module.set_scheduler(
                    optimizers[0], self.scheduler_name, **self.scheduler_args
                )
            else:
                assert loaded_scheduler is not None
                scheduler_ = loaded_scheduler

        # Utils
        criterion = nn.CrossEntropyLoss()
        best_acc = 0.0 if loaded_best_acc == -1 else loaded_best_acc

        # Internal helper function on pesudo forward
        def _forward(estimators, data):
            outputs = [
                F.softmax(estimator(data), dim=1) for estimator in estimators
            ]
            proba = op.average(outputs)

            return proba

        # Maintain a pool of workers
        with Parallel(n_jobs=self.n_jobs) as parallel:

            # Training loop
            for epoch in np.array(list(range(epochs - loaded_epoch))) + loaded_epoch:
                self.train()

                if self.use_scheduler_:
                    cur_lr = scheduler_.get_last_lr()[0]
                else:
                    cur_lr = None

                if self.n_jobs and self.n_jobs > 1:
                    msg = "Parallelization on the training epoch: {:03d}"
                    self.logger.info(msg.format(epoch))

                rets = parallel(
                    delayed(_parallel_fit_per_epoch)(
                        train_loader,
                        estimator,
                        cur_lr,
                        optimizer,
                        criterion,
                        idx,
                        epoch,
                        log_interval,
                        self.device,
                        True,
                    )
                    for idx, (estimator, optimizer) in enumerate(
                        zip(estimators, optimizers)
                    )
                )

                estimators, optimizers = [], []
                for estimator, optimizer in rets:
                    estimators.append(estimator)
                    optimizers.append(optimizer)

                # Validation
                save_flag = False
                if test_loader:
                    self.eval()
                    with torch.no_grad():
                        correct = 0
                        total = 0
                        for _, (data, target) in enumerate(test_loader):
                            data = data.to(self.device)
                            target = target.to(self.device)
                            output = _forward(estimators, data)
                            _, predicted = torch.max(output.data, 1)
                            correct += (predicted == target).sum().item()
                            total += target.size(0)
                        acc = 100 * correct / total

                        if acc > best_acc:
                            best_acc = acc
                            save_flag = True
                            self.estimators_ = nn.ModuleList()
                            self.estimators_.extend(estimators)

                        msg = (
                            "Epoch: {:03d} | Validation Acc: {:.3f}"
                            " % | Historical Best: {:.3f} %"
                        )
                        self.logger.info(msg.format(epoch, acc, best_acc))
                        if self.tb_logger:
                            self.tb_logger.add_scalar(
                                "voting/Validation_Acc", acc, epoch
                            )

                # Update the scheduler
                with warnings.catch_warnings():

                    # UserWarning raised by PyTorch is ignored because
                    # scheduler does not have a real effect on the optimizer.
                    warnings.simplefilter("ignore", UserWarning)

                    if self.use_scheduler_:
                        scheduler_.step()

                if save_model and save_flag:
                    io.save(self, epoch + 1, optimizers, scheduler_, save_dir, self.logger, best_acc)

        self.estimators_ = nn.ModuleList()
        self.estimators_.extend(estimators)
        if save_model and not test_loader:
            io.save(self, epoch + 1, optimizers, scheduler_, save_dir, self.logger, best_acc)

    @torchensemble_model_doc(item="classifier_evaluate")
    def evaluate(self, test_loader, return_loss=False):
        return super().evaluate(test_loader, return_loss)

    @torchensemble_model_doc(item="predict")
    def predict(self, X, return_numpy=True):
        return super().predict(X, return_numpy)


@torchensemble_model_doc("""Implementation on the VotingRegressor.""", "model")
class VotingRegressor(BaseRegressor):
    @torchensemble_model_doc(
        """Implementation on the data forwarding in VotingRegressor.""",
        "regressor_forward",
    )
    def forward(self, x):
        # Average over predictions from all base estimators.
        outputs = [estimator(x) for estimator in self.estimators_]
        pred = op.average(outputs)

        return pred

    @torchensemble_model_doc(
        """Set the attributes on optimizer for VotingRegressor.""",
        "set_optimizer",
    )
    def set_optimizer(self, optimizer_name, **kwargs):
        super().set_optimizer(optimizer_name, **kwargs)

    @torchensemble_model_doc(
        """Set the attributes on scheduler for VotingRegressor.""",
        "set_scheduler",
    )
    def set_scheduler(self, scheduler_name, **kwargs):
        super().set_scheduler(scheduler_name, **kwargs)

    @torchensemble_model_doc(
        """Implementation on the training stage of VotingRegressor.""", "fit"
    )
    def fit(
        self,
        train_loader,
        epochs=100,
        log_interval=100,
        test_loader=None,
        save_model=True,
        save_dir=None,
    ):

        self._validate_parameters(epochs, log_interval)
        self.n_outputs = self._decide_n_outputs(train_loader)

        # Instantiate a pool of base estimators, optimizers, and schedulers.
        estimators = []
        for _ in range(self.n_estimators):
            estimators.append(self._make_estimator())

        optimizers = []
        for i in range(self.n_estimators):
            optimizers.append(
                set_module.set_optimizer(
                    estimators[i], self.optimizer_name, **self.optimizer_args
                )
            )

        if self.use_scheduler_:
            scheduler_ = set_module.set_scheduler(
                optimizers[0], self.scheduler_name, **self.scheduler_args
            )

        # Utils
        criterion = nn.MSELoss()
        best_mse = float("inf")

        # Internal helper function on pesudo forward
        def _forward(estimators, data):
            outputs = [estimator(data) for estimator in estimators]
            pred = op.average(outputs)

            return pred

        # Maintain a pool of workers
        with Parallel(n_jobs=self.n_jobs) as parallel:

            # Training loop
            for epoch in range(epochs):
                self.train()

                if self.use_scheduler_:
                    cur_lr = scheduler_.get_last_lr()[0]
                else:
                    cur_lr = None

                if self.n_jobs and self.n_jobs > 1:
                    msg = "Parallelization on the training epoch: {:03d}"
                    self.logger.info(msg.format(epoch))

                rets = parallel(
                    delayed(_parallel_fit_per_epoch)(
                        train_loader,
                        estimator,
                        cur_lr,
                        optimizer,
                        criterion,
                        idx,
                        epoch,
                        log_interval,
                        self.device,
                        False,
                    )
                    for idx, (estimator, optimizer) in enumerate(
                        zip(estimators, optimizers)
                    )
                )

                estimators, optimizers = [], []
                for estimator, optimizer in rets:
                    estimators.append(estimator)
                    optimizers.append(optimizer)

                # Validation
                if test_loader:
                    self.eval()
                    with torch.no_grad():
                        mse = 0.0
                        for _, (data, target) in enumerate(test_loader):
                            data = data.to(self.device)
                            target = target.to(self.device)
                            output = _forward(estimators, data)
                            mse += criterion(output, target)
                        mse /= len(test_loader)

                        if mse < best_mse:
                            best_mse = mse
                            self.estimators_ = nn.ModuleList()
                            self.estimators_.extend(estimators)
                            if save_model:
                                io.save(self, save_dir, self.logger)

                        msg = (
                            "Epoch: {:03d} | Validation MSE:"
                            " {:.5f} | Historical Best: {:.5f}"
                        )
                        self.logger.info(msg.format(epoch, mse, best_mse))
                        if self.tb_logger:
                            self.tb_logger.add_scalar(
                                "voting/Validation_MSE", mse, epoch
                            )

                # Update the scheduler
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)

                    if self.use_scheduler_:
                        scheduler_.step()

        self.estimators_ = nn.ModuleList()
        self.estimators_.extend(estimators)
        if save_model and not test_loader:
            io.save(self, save_dir, self.logger)

    @torchensemble_model_doc(item="regressor_evaluate")
    def evaluate(self, test_loader):
        return super().evaluate(test_loader)

    @torchensemble_model_doc(item="predict")
    def predict(self, X, return_numpy=True):
        return super().predict(X, return_numpy)
