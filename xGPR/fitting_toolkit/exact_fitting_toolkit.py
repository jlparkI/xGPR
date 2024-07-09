"""Implements functions for fitting the weights using exact calculations,
which is generally only recommended for small datasets & numbers of
random features, and for fitting the variance using exact calculations."""
try:
    import cupy as cp
    import cupyx as cpx
except:
    pass
import numpy as np
from scipy.linalg import cho_solve

from ..scoring_toolkit.exact_nmll_calcs import calc_var_design_mat
from ..scoring_toolkit.exact_nmll_calcs import direct_weight_calc, calc_design_mat



def calc_weights_exact(dataset, kernel):
    """Calculates the weights when fitting the model using
    matrix decomposition. Exact and fast for small numbers
    of random features but poor scaling.

    Args:
        dataset: Either OnlineDataset or OfflineDataset,
            containing the information on the dataset we
            are fitting.
        kernel: A valid kernel object that can generate random
            features.

    Returns:
        weights: A cupy or numpy array of shape (M) for M
            random features.
    """
    z_trans_z, z_trans_y, _ = calc_design_mat(dataset, kernel)
    lambda_p = kernel.get_hyperparams(logspace=False)[0]
    z_trans_z.flat[::z_trans_z.shape[0]+1] += lambda_p**2
    _, weights = direct_weight_calc(z_trans_z, z_trans_y, kernel)
    return weights, 1, []



def calc_variance_exact(kernel, dataset, kernel_choice, variance_rffs):
    """Calculates the var matrix used for calculating
    posterior predictive variance on new datapoints. We
    only ever use closed-form matrix-decomposition based
    calculations here since the variance does not need to
    be approximated as accurately as the posterior predictive
    mean, so we can restrict the user to a smaller number of
    random features (defined in constants.constants).

    Args:
        kernel: A valid kernel object that can generate random features.
        dataset: Either an OnlineDataset or an OfflineDataset containing
            the data that needs to be fitted.
        kernel_choice (str): The type of kernel.
        variance_rffs (int): The number of random features for variance.

    Returns:
        var: A cupy or numpy array of shape (M, M) where M is the
            number of random features.
    """
    z_trans_z = calc_var_design_mat(dataset, kernel,
                        variance_rffs)
    lambda_ = kernel.get_lambda()
    z_trans_z.flat[::z_trans_z.shape[0]+1] += lambda_**2
    if kernel.device == "cpu":
        var = np.linalg.pinv(z_trans_z)
    else:
        var = cp.linalg.pinv(z_trans_z)
    return var


def calc_discriminant_weights_exact(dataset, kernel, x_mean,
        targets):
    """Calculates the weights when fitting a discriminant using
    exact matrix decomposition. Fast for small numbers of random
    features but poor scaling.

    Args:
        dataset: Either OnlineDataset or OfflineDataset,
            containing the information on the dataset we
            are fitting.
        kernel: A valid kernel object that can generate random
            features.
        x_mean (ndarray): An array of shape (num_rffs) containing
            the mean of the training data.
        targets (ndarray): A (num_rffs, nc) for nc classes shape
            array containing class specific means.


    Returns:
        weights: A cupy or numpy array of shape (M, nc) for M
            random features and nc classes.
    """
    num_rffs = kernel.get_num_rffs()
    if kernel.device == "cpu":
        z_trans_z = np.zeros((num_rffs, num_rffs))
    else:
        z_trans_z = cp.zeros((num_rffs, num_rffs))

    for i, (xdata, ldata) in enumerate(dataset.get_chunked_x_data()):
        xfeatures = kernel.transform_x(xdata, ldata) - x_mean[None,:]
        z_trans_z += xfeatures.T @ xfeatures
        if i % 2 == 0:
            if kernel.device == "cuda":
                mempool = cp.get_default_memory_pool()
                mempool.free_all_blocks()

    z_trans_z /= float(dataset.get_ndatapoints())
    z_trans_z.flat[::z_trans_z.shape[0]+1] += kernel.get_lambda()**2

    if kernel.device == "cpu":
        chol_z_trans_z = np.linalg.cholesky(z_trans_z)
        weights = cho_solve((chol_z_trans_z, True), targets)
    else:
        chol_z_trans_z = cp.linalg.cholesky(z_trans_z)
        weights = cpx.scipy.linalg.solve_triangular(chol_z_trans_z,
                        targets, lower=True)
        weights = cpx.scipy.linalg.solve_triangular(chol_z_trans_z.T,
                        weights, lower=False)
    return weights
