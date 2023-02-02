"""Provides a toolkit for performing optimization using
cross-validations or performance on a validation set. Can be
scaled to large numbers of random features."""
import copy
import numpy as np
from scipy.optimize import minimize

from .hparam_scoring import _get_cv_score



def _check_inputs(max_feval, model,
                    preset_hparams, score_type):
    """Checks the inputs for likely problems that would cause a poor result."""
    if max_feval < 10 or max_feval > 250:
        raise ValueError("Max feval must be >= 10 and <= 250.")
    if model.kernel is None:
        if preset_hparams is None:
            raise ValueError("Either the model should already have been tuned, "
                        "or preset_hparams should be supplied.")
        elif preset_hparams.shape[0] > 4:
            raise ValueError("The Bayesian CV optimizer is only applicable for kernels with 4 "
                "or fewer hyperparameters.")
    elif model.kernel.get_hyperparams().shape[0] > 4:
        raise ValueError("The Bayesian CV optimizer is only applicable for kernels with 4 "
                "or fewer hyperparameters.")
    if score_type not in ["mae", "mse"]:
        raise ValueError("Currently allowed score types are 'mae', 'mse'.")


def Direct_Fitting_Optimizer(model, train_dset,
                    bounds, optim_method = "Nelder-Mead",
                    max_feval = 25,
                    validation_dset = None,
                    preset_hparams = None,
                    random_state = 123,
                    score_type = "mae",
                    tol = 1e-3,
                    cg_tol = 1e-6,
                    verbose = True,
                    pretransform_dir = None,
                    mode = "cg"):
    """Conducts a direct hyperparameter optimization using validation
    set performance to score.

    Args:
        model: A valid xGP_Regression model object.
        train_dset: Either an OnlineDataset or OfflineDataset containing raw
            data.
        bounds (np.ndarray): A numpy array with the same number of rows
            as the specified kernel has hyperparameters, where column 0 is
            the low bound and column 1 is the high.
        optim_method (str): One of "Powell", "Nelder-Mead". Determines
            how optimization is conducted.
        max_feval (int): The maximum number of iterations to run.
        validation_dset: Either None or a valid OnlineDataset or OfflineDataset.
            If None, 5x cross-validations are conducted on train_dset. Otherwise,
            the scoring is done on validation_dset, train_dset is used for fitting
            only.
        preset_hparams: Either None or an np.ndarray of the same length as the number
            of hyperparameters for the kernel used by 'model'. If None, the model
            must have already been tuned so that it has an existing set of hyperparameters
            that will be used as a starting point.
        random_state (int): A seed for the random number generator.
        score_type (str): One of 'mae', 'mse'. If 'mae', mean absolute error
            is optimized. If 'mse', mean square error is optimized.
        tol (float): The threshold for convergence.
        cg_tol (float): The threshold for conjugate gradients convergence.
        verbose (bool): If True, regular updates are printed.
        pretransform_dir (str): Either None or a valid filepath where "pretransformed"
            data can be saved. This may provide some speedup on an SSD.

    Raises:
        ValueError: A ValueError is raised if the inputs are invalid.

    Returns:
        best_hparams (np.ndarray): A new set of optimized hyperparameters.
    """
    _check_inputs(max_feval, model, preset_hparams, score_type)

    original_verbosity = copy.deepcopy(model.verbose)
    model.verbose = False

    if preset_hparams is not None:
        hparams = preset_hparams
    elif model.kernel is not None:
        hparams = model.kernel.get_hyperparams(logspace = True)
    else:
        raise ValueError("Either the model should already have been tuned, "
                "or a starting point should be supplied.")

    res = minimize(_get_cv_score, x0 = hparams,
                args=(model, train_dset, validation_dset,
                    random_state, score_type, verbose,
                    pretransform_dir, mode, cg_tol),
                    method=optim_method,
                    options = {"maxfev":max_feval,
                        "xatol":tol, "fatol":tol,
                        "xtol":tol, "ftol":tol},
                    bounds = bounds)

    model.verbose = original_verbosity
    model.kernel.set_hyperparams(res.x)
    return res.x
