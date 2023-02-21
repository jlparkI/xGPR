"""Implements functions for fitting the weights using exact calculations,
which is generally only recommended for small datasets & numbers of
random features, and for fitting the variance using exact calculations."""
try:
    import cupy as cp
except:
    pass
import numpy as np

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
    #This is a very naughty hack.
    #TODO: Add a proper variance calc for linear.
    if kernel_choice == "Linear":
        return None
    z_trans_z = calc_var_design_mat(dataset, kernel,
                        variance_rffs)
    lambda_ = kernel.get_lambda()
    z_trans_z.flat[::z_trans_z.shape[0]+1] += lambda_**2
    if kernel.device == "cpu":
        var = np.linalg.pinv(z_trans_z)
    else:
        var = cp.linalg.pinv(z_trans_z)
    return var
