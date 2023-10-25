"""Describes the bayes_grid optimization routine, which uses a combination
of accelerated gridsearch (to reduce 4d and 3d problems to 2d and 1d)
with Bayesian optimization to find good hyperparameters."""
import warnings

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, Matern

from .lb_optimizer import shared_hparam_search

def bayes_grid_tuning(kernel, dataset, bounds, random_seed,
                    max_iter, verbose, tol = 1e-1,
                    n_pts_per_dim = 10, n_cycles = 3,
                    n_init_pts = 10,
                    subsample = 1,
                    eigval_quotient = 1e6,
                    min_eigval = 1e-6):
    """Conducts accelerated gridsearch optimization + Bayesian
    optimization.

    Args:
        kernel: A valid kernel that can generate random features.
        dataset: An OnlineDataset or OfflineDataset containing raw data.
        bounds (np.ndarray): An N x 2 for N hyperparameters set of boundaries
            for optimization.
        random_seed (int): A seed for the random number generator.
        max_iter (int): The maximum number of iterations.
        verbose (bool): If True, print regular updates.
        tol (float): The threshold for convergence.
        n_pts_per_dim (int): The number of grid points per shared hparam.
        n_cycles (int): The number of cycles of "telescoping" grid search
            to run.
        n_init_pts (int): The number of initial grid points to evaluate before
            Bayesian optimization. 10 (the default) is usually fine. If you are
            searcing a smaller space, however, you can sometimes save time by using
            a smaller # (e.g. 5).
        subsample (float): A value in the range [0.01,1] that indicates what
            fraction of the training set to use each time the gradient is
            calculated (the same subset is used every time). In general, 1
            will give better results, but using a subsampled subset can be
            a fast way to find the (approximate) location of a good
            hyperparameter set.
        eigval_quotient (float): A value by which the largest singular value
            of Z^T Z is divided to determine the smallest acceptable eigenvalue
            of Z^T Z (singular vectors with eigenvalues smaller than this
            are discarded). Setting this to larger values will make crude_bayes
            slightly more accurate, at the risk of numerical stability issues.
            In general, do not change this without good reason.
        min_eigval (float): If the largest singular value of Z^T Z divided by
            eigval_quotient is < min_eigval, min_eigval is used as the cutoff
            threshold instead. Setting this to smaller values will make crude_
            bayes slightly more accurate, at the risk of numerical stability
            issues. In general, do not change this without good reason.

    Raises:
        ValueError: A ValueError is raised if this is run with a kernel with >
            5 or <= 2 hyperparameters.
    """

    if bounds.shape[0] >= 5 or bounds.shape[0] < 3:
        raise ValueError("Bayesian optimization is only allowed for kernels with "
                "3 - 4 hyperparameters.")


    if bounds.shape[0] == 3:
        sigma_grid,_ = get_grid_pts(n_init_pts, bounds)
    else:
        sigma_grid = get_random_starting_pts(n_init_pts, bounds, random_seed)

    sigma_grid = np.round(sigma_grid, 7)

    lb_vals, scores = [], []
    if len(sigma_grid.shape) == 1:
        sigma_grid = sigma_grid.reshape(-1,1)
    sigma_grid = list(sigma_grid)

    for i, sigma_pt in enumerate(sigma_grid):
        score, lb_val = shared_hparam_search(sigma_pt, kernel, dataset, bounds[:2,:],
                        n_pts_per_dim = n_pts_per_dim, n_cycles = n_cycles,
                        subsample = subsample, eigval_quotient = eigval_quotient,
                        min_eigval = min_eigval)

        scores.append(score)
        lb_vals.append(lb_val)
        if verbose:
            print(f"Grid point {i} acquired.")

    scores = np.asarray(scores)
    smallest_non_inf_val = np.max(scores[scores < np.inf])
    scores[scores == np.inf] = smallest_non_inf_val
    scores = scores.tolist()
    surrogate = GPR(kernel = RBF(),
            normalize_y = True,
            alpha = 1e-6, random_state = random_seed,
            n_restarts_optimizer = 4)


    sigma_bounds = bounds[2:, :]
    iternum = len(sigma_grid)
    for iternum in range(len(sigma_grid), max_iter):
        new_sigma, min_dist, surrogate = propose_new_point(
                            sigma_grid, scores, surrogate,
                            sigma_bounds, random_seed + iternum)
        if verbose:
            print(f"New hparams: {new_sigma}")
        score, lb_val = shared_hparam_search(new_sigma, kernel, dataset, bounds[:2,:],
                        n_pts_per_dim = n_pts_per_dim, n_cycles = n_cycles,
                        subsample = subsample, eigval_quotient = eigval_quotient,
                        min_eigval = min_eigval)
        sigma_grid.append(new_sigma)
        lb_vals.append(lb_val)
        scores.append(min(score, smallest_non_inf_val))

        if iternum > 15 and min_dist < tol:
            break
        if verbose:
            print(f"Additional acquisition {iternum}.")

    best_hparams = np.empty((bounds.shape[0]))
    best_hparams[2:] = sigma_grid[np.argmin(scores)]
    best_hparams[:2]  = lb_vals[np.argmin(scores)]
    if verbose:
        print(f"Best score achieved: {np.round(np.min(scores), 4)}")
        print(f"Best hyperparams: {best_hparams}")
    return best_hparams, (sigma_grid, scores), np.min(scores), iternum


def propose_new_point(sigma_vals, scores,
                surrogate, bounds, random_seed,
                num_cand = 500):
    """Refits the 'surrogate' model and uses it to propose new
    locations for exploration (via Thompson sampling).

    Args:
        sigma_vals (np.ndarray): A grid of kernel-specific hyperparameters
            at which NMLL has been evaluated so far.
        scores (array-like): The scores for sigma_vals.
        surrogate: A scikit-learn Gaussian process model with a Matern kernel.
            Is fitted to sigma_vals, scores.
        bounds (np.ndarray): An N x 2 for N kernel specific hyperparameters
            array of boundaries for optimization.
        random_seed (int): A seed for the random number generator.
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
        xvals = np.vstack(sigma_vals)
        surrogate.fit(xvals, scores)

    rng = np.random.default_rng(random_seed)
    candidates = rng.uniform(low=bounds[:,0], high=bounds[:,1],
                           size = (num_cand, bounds.shape[0]))
    candidates = np.round(candidates, 7)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_candidates = surrogate.sample_y(candidates, n_samples=15,
                    random_state=random_seed)
    best_idx = np.unravel_index(y_candidates.argmin(),
                            y_candidates.shape)
    best_cand = candidates[best_idx[0],:]
    min_dist = np.min(np.linalg.norm(best_cand[None,:] - xvals, axis=1))
    return candidates[best_idx[0],:], min_dist, surrogate


def get_random_starting_pts(num_sigma_vals, bounds, random_seed = 123):
    """For kernels with 4 or 5 hyperparameters, it is often more efficient
    to randomly populate the search space with an initial number of search
    points, and then build on these with Bayesian optimization. This function
    generates the lambda_beta grid plus random 'exploration' values for
    the kernel-specific hyperparameters (aka 'sigma').

    Args:
        num_sigma_vals (int): The number of kernel-specific hyperparameter
            combinations to sample.
        bounds (np.ndarray): The boundaries of the search space.
        random_seed (int): A seed to the random number generator.

    Returns:
        sigma_grid (np.ndarray): The initial kernel-specific hyperparameter combinations to
            evaluate.
        lambda_beta_grid (np.ndarray): The shared hyperparameter combinations to evaluate for each
            sigma combination.
    """
    rng = np.random.default_rng(random_seed)
    sigma_grid = np.empty((num_sigma_vals, bounds.shape[0] - 2))
    for i in range(sigma_grid.shape[1]):
        sigma_grid[:,i] = rng.uniform(size = num_sigma_vals,
                        low = bounds[i + 2, 0], high = bounds[i + 2, 1])
    return sigma_grid


def get_grid_pts(num_pts_per_sigma, bounds):
    """Builds a starting grid of sigma points (kernel-specific
    hyperparameters).
    Args:
        num_pts_per_sigma (int): The number of points at which to
            evaluate NMLL per kernel-specific hyperparameter.
        bounds (np.ndarray): A numpy array of shape N x 2 for N
            hyperparameters indicating the boundaries within which
            to optimize.
    Returns:
        sigma_pts: Either [None] or a numpy array of shape
            (num_pts_per_axis, C) where C is the number of kernel-
            specific hyperparameters. NMLL is evaluated at each
            row in sigma_pts.
    """
    if bounds.shape[0] == 2:
        sigma_pts = [None]
    elif bounds.shape[0] == 3:
        sigma_pts = np.linspace(bounds[2,0], bounds[2,1],
                                num_pts_per_sigma)
    elif bounds.shape[0] == 4:
        sigma1 = np.linspace(bounds[2,0], bounds[2,1],
                                num_pts_per_sigma)
        sigma2 = np.linspace(bounds[3,0], bounds[3,1],
                                num_pts_per_sigma)
        sigma1, sigma2 = np.meshgrid(sigma1, sigma2)
        sigma_pts = np.array((sigma1.ravel(), sigma2.ravel())).T
    else:
        raise ValueError("Grid search is only applicable for "
                    "kernels with <= 4 hyperparameters.")
    scoregrid = np.zeros((len(sigma_pts)))
    return sigma_pts, scoregrid
