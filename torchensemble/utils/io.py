import os
import torch

from torchensemble.utils import set_module


def save(model, epoch, optimizers, scheduler, save_dir, logger, best_acc=-1, est_idx=-1):
    """Implement model serialization to the specified directory."""
    if save_dir is None:
        save_dir = "./"

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    # Decide the base estimator name
    if isinstance(model.base_estimator_, type):
        base_estimator_name = model.base_estimator_.__name__
    else:
        base_estimator_name = model.base_estimator_.__class__.__name__

    # {Ensemble_Model_Name}_{Base_Estimator_Name}_{n_estimators}
    filename = "{}_{}_{}_ckpt.pth".format(
        type(model).__name__,
        base_estimator_name,
        model.n_estimators,
    )

    # The real number of base estimators in some ensembles is not same as
    # `n_estimators`.
    state = {
        "n_estimators": len(model.estimators_),
        "model": model.state_dict(),
        "optimizers": [opt.state_dict() for opt in optimizers],
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "est_idx": est_idx,
        "best_acc": best_acc,
    }
    save_dir = os.path.join(save_dir, filename)

    logger.info("Saving the model to `{}`".format(save_dir))

    # Save
    torch.save(state, save_dir)

    return


def load(model, save_dir="./", logger=None, use_scheduler=False, ):
    """Implement model deserialization from the specified directory."""
    if not os.path.exists(save_dir):
        raise FileExistsError("`{}` does not exist".format(save_dir))

    # Decide the base estimator name
    if isinstance(model.base_estimator_, type):
        base_estimator_name = model.base_estimator_.__name__
    else:
        base_estimator_name = model.base_estimator_.__class__.__name__

    # {Ensemble_Model_Name}_{Base_Estimator_Name}_{n_estimators}
    filename = "{}_{}_{}_ckpt.pth".format(
        type(model).__name__,
        base_estimator_name,
        model.n_estimators,
    )
    save_dir = os.path.join(save_dir, filename)

    if logger:
        logger.info("Loading the model from `{}`".format(save_dir))

    state = torch.load(save_dir)
    n_estimators = state["n_estimators"]
    model_params = state["model"]
    optimizers_params = state["optimizers"]
    scheduler_params = state["scheduler"]
    epoch = state["epoch"]
    est_idx = state["est_idx"]
    best_acc = state["best_acc"]

    # Pre-allocate and load all base estimators
    for _ in range(n_estimators):
        model.estimators_.append(model._make_estimator())
    model.load_state_dict(model_params)

    if len(optimizers_params) == 1:
        if est_idx == -1:  # Fusion or SoftGradientBoosting
            optimizers = []
            optimizers.append(
                set_module.set_optimizer(
                    model, model.optimizer_name, **model.optimizer_args  # "model" as an input
                )
            )
        else:  # GradientBoosting (optimizer and scheduler are not used actually)
            optimizers = []
            optimizers.append(
                set_module.set_optimizer(
                    model.estimators_[est_idx], model.optimizer_name, **model.optimizer_args  # estimators[est_idx] as an input
                )
            )
        optimizers[0].load_state_dict(optimizers_params[0])
    else:
        optimizers = []
        for i in range(len(optimizers_params)):
            optimizers.append(
                set_module.set_optimizer(
                    model.estimators_[i], model.optimizer_name, **model.optimizer_args
                )
            )
        for i in range(len(optimizers_params)):
            optimizers[i].load_state_dict(optimizers_params[i])

    if use_scheduler:
        scheduler_ = set_module.set_scheduler(
            optimizers[0], model.scheduler_name, **model.scheduler_args
        )
        scheduler_.load_state_dict(scheduler_params)
        return epoch, optimizers, scheduler_, best_acc, est_idx

    else:
        return epoch, optimizers, best_acc, est_idx
