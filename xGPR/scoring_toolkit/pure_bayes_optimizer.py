"""Bayesian optimization for hyperparameters without the
efficient strategy for shared hyperparameters used in bayes_grid."""
import sys
import copy
import warnings
import numpy as np
from scipy.stats import qmc
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import Matern, RBF
from ..constants import constants


def pure_bayes_tuning(cost_function, bounds, random_seed,
        max_iter, verbose = True, tol = 1e-1, cost_args = None):
    """Bayesian optimization for hyperparameters, without gradient
    information. TODO: We can make this much more efficient if
    we add gradient information most likely.

    Args:
        cost_function: A function that can calculate the negative
            marginal log likelihood (aka 'score') using the provided
            dataset.
        tuning_dataset: An object of type Dataset, either offline or
            online etc. as applicable.
        bounds (np.ndarray): An n x 2 array where n is the number of
            hyperparameters setting boundaries for optimization.
        random_seed (int): A seed for the random number generator.
        max_iter (int): The maximum number of iterations.
        verbose (bool): If True, regular updates are printed.
        tol (float): The criteria for convergence.
        cost_args (tuple): A tuple of arguments to be passed to the cost function.

    Returns:
        hyperparams (np.ndarray): A shape-n array containing the hyperparameters
            id'd as best during optimization.
        best_score (float): The best nmll from optimization.
        n_feval (int): The number of iterations actually performed.
    """
    lhc_sampler = qmc.LatinHypercube(d = bounds.shape[0],
                            seed = random_seed)
    hparam_vals = lhc_sampler.random(n=10)
    hparam_vals = np.round(qmc.scale(hparam_vals, bounds[:,0],
                        bounds[:,1]), 1)
    scores = []
    hparam_vals = list(hparam_vals)

    for i, hparam_val in enumerate(hparam_vals):
        scores.append(cost_function(hparam_val, *cost_args))
        if verbose:
            print(f"Point {i} acquired.")

    scores = adjust_problem_scores(scores)
    surrogate = GPR(kernel = RBF(),#Matern(nu=5/2),
            normalize_y = True,
            alpha = 1e-6, random_state = random_seed,
            n_restarts_optimizer = 5)

    for iternum in range(len(hparam_vals), max_iter):
        new_hparam, min_dist, surrogate = propose_new_point(
                            hparam_vals, scores, surrogate,
                            bounds, random_seed + iternum,
                            itnum=iternum)
        score = cost_function(new_hparam, *cost_args)
        hparam_vals.append(new_hparam)
        scores.append(min(score, np.max(scores)))

        if min_dist < tol:
            break
        if verbose:
            print(f"Additional acquisition {iternum}.")

    best_hparams = hparam_vals[np.argmin(scores)]
    if verbose:
        print(f"Best score achieved: {np.min(scores)}")
    return best_hparams, np.min(scores), iternum




def propose_new_point(hparam_vals, scores,
                surrogate, bounds, random_state,
                num_cand = 500, itnum=0):
    """Refits the 'surrogate' model and uses it to propose new
    locations for exploration (via Thompson sampling).

    Args:
        hparam_vals (np.ndarray): A set of hyperparameters
            at which NMLL has been evaluated so far.
        scores (array-like): The scores for hparam_vals.
        surrogate: A scikit-learn Gaussian process model with an RBF kernel.
            Is fitted to hparam_vals, scores.
        bounds (np.ndarray): An N x 2 for N kernel specific hyperparameters
            array of boundaries for optimization.
        random_state (int): A seed for the random number generator.
        num_cand (int): The number of candidate points to evaluate.

    Returns:
        best_candidate (np.ndarray): A set of kernel-specific hyperparameters
            at which to propose the next acquisition.
        min_dist (float): The smallest distance between best_candidate
            and any existing candidates.
        surrogate: The Gaussian process acquisition model now refitted
            to the updated data.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xvals = np.round(np.vstack(hparam_vals), 1)
        surrogate.fit(xvals, scores)

    rng = np.random.default_rng(random_state)
    candidates = np.round(rng.uniform(low=bounds[:,0], high=bounds[:,1],
                           size = (num_cand, bounds.shape[0])), 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_candidates = surrogate.sample_y(candidates, n_samples=50,
                    random_state=random_state)
    best_idx = np.unravel_index(y_candidates.argmin(),
                            y_candidates.shape)
    best_cand = candidates[best_idx[0],:]
    min_dist = np.min(np.linalg.norm(best_cand[None,:] - xvals,
                    axis=1))
    return best_cand, min_dist, surrogate


def adjust_problem_scores(scores):
    """The scoring function assigns some arbitrarily high value if a problem is
    encountered. Set such points to the lowest NON-problem score to avoid misleading
    the GP.

    Args:
        scores (list): Scores acquired to date.

    Returns:
        scores (list): The scores with any objectionable values corrected.
    """
    scores = np.asarray(scores)
    largest_acceptable_val = np.max(scores[scores < constants.DEFAULT_SCORE_IF_PROBLEM])
    scores[scores == constants.DEFAULT_SCORE_IF_PROBLEM] = largest_acceptable_val
    scores = scores.tolist()
    return scores

def _ucb(x, surrogate):
    """Calculates the lower confidence bound."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mean, std = surrogate.predict(x.reshape(1,-1), return_std=True)

    return mean - 1.96 * std

def _ei(x, gp, y_max):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if len(x.shape) == 1:
            mean, std = gp.predict(x.reshape(1,-1), return_std=True)
        else:
            mean, std = gp.predict(x, return_std=True)

    a = (mean - y_max)
    z = a / std
    return -(a * norm.cdf(z) + std * norm.pdf(z))
