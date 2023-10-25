"""Describes the crude_grid optimization routine, which uses
accelerated gridsearch (to reduce 3d problems to 1d)
to find good hyperparameters."""
import warnings

import numpy as np

from .lb_optimizer import shared_hparam_search
from .bayes_grid_optimizer import get_grid_pts

def crude_grid_tuning(kernel, dataset, init_bounds,
                    verbose, n_gridpoints = 30,
                    n_pts_per_dim = 10, subsample = 1,
                    eigval_quotient = 1e6,
                    min_eigval = 1e-6):
    """Conducts accelerated gridsearch optimization.

    Args:
        kernel: A valid kernel that can generate random features.
        dataset: An OnlineDataset or OfflineDataset containing raw data.
        init_bounds (np.ndarray): An N x 2 for N hyperparameters set of boundaries
            for optimization.
        verbose (bool): If True, print regular updates.
        n_gridpoints (int): The number of gridpoints.
        n_pts_per_dim (int): The number of grid points per shared hparam.
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
            3 or <= 2 hyperparameters.
    """
    bounds = init_bounds.copy()
    if bounds.shape[0] != 3:
        raise ValueError("crude_grid is only allowed for kernels with 3 hyperparameters.")


    sigma_grid,_ = get_grid_pts(n_gridpoints, bounds)
    sigma_grid = np.round(sigma_grid, 7)

    sigma_grid = sigma_grid.reshape(-1,1)
    sigma_grid = list(sigma_grid)
    lb_vals, scores = [], []

    for i, sigma_pt in enumerate(sigma_grid):
        score, lb_val = shared_hparam_search(sigma_pt, kernel, dataset, bounds[:2,:],
                        n_pts_per_dim = n_pts_per_dim, n_cycles = 3,
                        subsample = subsample,
                        eigval_quotient = eigval_quotient,
                        min_eigval = min_eigval)

        scores.append(score)
        lb_vals.append(lb_val)
        if verbose:
            print(f"Grid point {i} acquired.")

    scores = np.asarray(scores)
    smallest_non_inf_val = np.max(scores[scores < np.inf])
    scores[scores == np.inf] = smallest_non_inf_val
    scores = scores.tolist()

    best_hparams = np.empty((bounds.shape[0]))
    best_hparams[2:] = sigma_grid[np.argmin(scores)]
    best_hparams[:2]  = lb_vals[np.argmin(scores)]
    if verbose:
        print(f"Best score achieved: {np.round(np.min(scores), 4)}")
        print(f"Best hyperparams: {best_hparams}")
    return best_hparams, (sigma_grid, scores), np.min(scores)
