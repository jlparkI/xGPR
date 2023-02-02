"""Provides a toolkit for performing Bayesian optimization using
cross-validations or performance on a validation set. Can be
scaled to large numbers of random features."""
import copy
import time
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import Matern
import numpy as np

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


def Bayesian_Fitting_Optimizer(model, train_dset, bounds,
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
    """Conducts a Bayesian hyperparameter optimization using cross-validations
    to score.

    Args:
        model: A valid xGP_Regression model object.
        train_dset: Either an OnlineDataset or OfflineDataset containing raw
            data.
        bounds (np.ndarray): A numpy array with the same number of rows
            as the specified kernel has hyperparameters, where column 0 is
            the low bound and column 1 is the high.
        max_feval (int): The maximum number of function evaluations to run.
            Fitting terminates either once max_feval is reached OR the process
            converges, whichever happens first.
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
            is optimized. If 'mse', the mean square error is optimized.
        tol (float): The threshold for convergence.
        cg_tol (float): The threshold for conjugate gradients convergence.
        verbose (bool): If True, regular updates are provided.
        pretransform_dir (str): Either None or a valid filepath where "pretransformed"
            data can be saved. This may provide some speedup on an SSD.
        mode (str): One of "cg", "exact". If "exact", exact fitting is used so no
            preconditioner construction is required. "exact" will raise an error
            if the number of fitting RFFs is > the allowed for the selected kernel.

    Raises:
        ValueError: A ValueError is raised if the inputs are invalid.

    Returns:
        best_hparams (np.ndarray): A new set of optimized hyperparameters.
        cv_hparams (list): All of the hyperparameters evaluated during
            tuning.
        cv_scores (list): All of the scores retrieved during tuning.
    """
    _check_inputs(max_feval, model, preset_hparams, score_type)

    original_verbosity = copy.deepcopy(model.verbose)
    model.verbose = False
    model.variance_rffs = 16

    if preset_hparams is not None:
        hparams = preset_hparams
    elif model.kernel is not None:
        hparams = model.kernel.get_hyperparams(logspace = True)
    else:
        raise ValueError("Either the model should already have been tuned, "
                "or a starting point should be supplied.")

    cv_hparams = _get_hparam_samples(bounds, n_samples = 9,
                        random_state = random_state)
    cv_hparams = np.vstack([hparams[None,:], cv_hparams])
    cv_scores = [_get_cv_score(cv_hparams[i,:], model, train_dset,
                validation_dset, random_state, score_type,
                pretransform_dir = pretransform_dir, mode=mode, cg_tol = cg_tol)
                        for i in range(cv_hparams.shape[0])]
    cv_surrogate = GPR(kernel = Matern(nu=5/2),
            normalize_y = True,
            alpha = 1e-6, random_state = random_state,
            n_restarts_optimizer = 4)
    for i in range(10, max_feval):
        cv_surrogate.fit(cv_hparams, cv_scores)
        new_hparam, terminate = _propose_new_point(cv_hparams,
                            cv_surrogate, bounds, random_state + i, tol)
        if terminate:
            break
        cv_scores.append(_get_cv_score(new_hparam, model, train_dset,
                        validation_dset, random_state, score_type, verbose,
                        pretransform_dir = pretransform_dir, mode=mode, cg_tol = cg_tol))
        if verbose:
            print(f"Iteration {i} complete.")
        cv_hparams = np.append(cv_hparams, new_hparam[None,:],
                            axis=0)


    best_hparams = cv_hparams[np.argmin(cv_scores),:]

    model.verbose = original_verbosity
    if verbose:
        print(f"Best hparams: {best_hparams}")
    model.kernel.set_hyperparams(best_hparams)
    return best_hparams, cv_hparams, cv_scores



def _propose_new_point(cv_hparams, cv_surrogate,
                bounds, random_state,
                tol, num_cand = 500):
    """Proposes a new point for Bayesian optimization and determines
    whether the new point is sufficiently close to those already
    evaluated that convergence has been achieved.

    Args:
        cv_hparams (np.ndarray): The hyperparameters evaluated thus
            far.
        cv_surrogate (GPR): A Gaussian process model fitted to the
            hyperparameters and scores evaluated thus far.
        bounds (np.ndarray): The boundaries for optimization.
        random_state (int): A seed for the random number generator.
        tol (float): A threshold for convergence.
        num_cand (int): The number of new candidates to draw.

    Returns:
        candidates (np.ndarray): Either None, in which case convergence
            has been achieved, or an np.ndarray with a new set of
            hyperparameters to evaluate.
        converged (bool): Either True for convergence achieved or False
            otherwise.
    """
    rng = np.random.default_rng(random_state)
    candidates = rng.uniform(low=bounds[:,0], high=bounds[:,1],
                           size = (num_cand, bounds.shape[0]))
    y_candidates = cv_surrogate.sample_y(candidates, n_samples=10,
                    random_state=random_state)
    best_idx = np.unravel_index(y_candidates.argmin(),
                            y_candidates.shape)
    best_cand = candidates[best_idx[0],:]
    min_dist = np.min(np.linalg.norm(best_cand[None,:] - cv_hparams,
                    axis=1))
    if min_dist < tol:
        print("Terminating early; convergence achieved.")
        return None, True
    return candidates[best_idx[0],:], False


def _get_hparam_samples(grid_bounds, n_samples, random_state):
    """Generates a batch of sample hyperparameters of size n_samples
    within grid_bounds as points to start with for Bayesian optimization."""
    rng = np.random.default_rng(random_state)
    candidates = rng.uniform(low=grid_bounds[:,0], high=grid_bounds[:,1],
                           size = (n_samples, grid_bounds.shape[0]))
    return candidates
