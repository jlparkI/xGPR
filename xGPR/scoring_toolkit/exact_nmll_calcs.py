"""Provides a set of functions for computing exact NMLL,
exact NMLL gradients, and matrices and vectors useful
for these tasks. Some of these tools can also be used
for (exact) fitting routines."""
try:
    import cupy as cp
    import cupyx as cpx
except:
    pass
import numpy as np
from scipy.linalg import cho_solve

def calc_zty(dataset, kernel):
    """Calculates the vector Z^T y.

    Args:
        dataset: An Dataset object that can supply
            chunked data.
        kernel: A valid kernel object that can generate
            random features.
        device (str): One of "cpu", "gpu".

    Returns:
        z_trans_y (array): A shape (num_rffs)
            array that contains Z^T y.
        y_trans_y (float): The value y^T y.
    """
    if kernel.device == "gpu":
        z_trans_y = cp.zeros((kernel.get_num_rffs()))
    else:
        z_trans_y = np.zeros((kernel.get_num_rffs()))

    y_trans_y = 0

    if dataset.pretransformed:
        for xdata, ydata in dataset.get_chunked_data():
            z_trans_y += xdata.T @ ydata
            y_trans_y += float( (ydata**2).sum() )
    else:
        for xdata, ydata in dataset.get_chunked_data():
            zdata = kernel.transform_x(xdata)
            z_trans_y += zdata.T @ ydata
            y_trans_y += float( (ydata**2).sum() )
    return z_trans_y, y_trans_y


def calc_design_mat(dataset, kernel):
    """Calculates the z_trans_z (z^T z) matrix where Z is the random
    features generated from raw input data X. Also generates
    y^T y and z^T y.

    Args:
        dataset: An OnlineDataset or OfflineDataset object storing
            the data we will use.
        kernel: A valid kernel object that can generate
            random features.

    Returns:
        z_trans_z: The cupy or numpy matrix resulting from z^T z. Will
            be shape M x M for M random features.
        z_trans_y: The cupy or numpy length M array resulting from
            z^T y for M random features.
        y_trans_y (float): The result of the dot product of y with itself.
            Used for some marginal likelihood calculations.
    """
    num_rffs = kernel.get_num_rffs()
    if kernel.device == "cpu":
        z_trans_z, z_trans_y = np.zeros((num_rffs, num_rffs)), np.zeros((num_rffs))
    else:
        z_trans_z = cp.zeros((num_rffs, num_rffs))
        z_trans_y = cp.zeros((num_rffs))
    y_trans_y = 0.0
    if dataset.pretransformed:
        for xfeatures, ydata in dataset.get_chunked_data():
            z_trans_y += xfeatures.T @ ydata
            z_trans_z += xfeatures.T @ xfeatures
            y_trans_y += ydata.T @ ydata
    else:
        for xdata, ydata in dataset.get_chunked_data():
            xfeatures = kernel.transform_x(xdata)
            z_trans_y += xfeatures.T @ ydata
            z_trans_z += xfeatures.T @ xfeatures
            y_trans_y += ydata.T @ ydata
    return z_trans_z, z_trans_y, float(y_trans_y)



def direct_weight_calc(chol_z_trans_z, z_trans_y, kernel):
    """Calculates the cholesky decomposition of (z^T z + lambda)^-1
    and then uses this to calculate the weights as (z^T z + lambda)^-1 z^T y.
    This exact calculation is only suitable for < 10,000 random features or so;
    cholesky has O(M^3) scaling.

    Args:
        z_trans_z: An M x M cupy or numpy matrix where M is the number of
            random features formed from z^T z when z is the random features
            generated for raw input data X.
        z_trans_y: A length M cupy or numpy array where M is the number of
            random features, formed from z^T y.
        kernel: A valid kernel object that can generate random features.

    Returns:
        chol_z_trans_z: The cholesky decomposition of z_trans_z. An M x M
            cupy or numpy matrix.
        weights: A length M cupy or numpy array containing the weights.
    """
    lambda_p = kernel.get_hyperparams(logspace=False)[0]
    chol_z_trans_z.flat[::chol_z_trans_z.shape[0]+1] += lambda_p**2
    if kernel.device == "cpu":
        chol_z_trans_z = np.linalg.cholesky(chol_z_trans_z)
        weights = cho_solve((chol_z_trans_z, True), z_trans_y)
    else:
        chol_z_trans_z = cp.linalg.cholesky(chol_z_trans_z)
        weights = cpx.scipy.linalg.solve_triangular(chol_z_trans_z,
                        z_trans_y, lower=True)
        weights = cpx.scipy.linalg.solve_triangular(chol_z_trans_z.T,
                        weights, lower=False)
    return chol_z_trans_z, weights



def calc_var_design_mat(dataset, kernel, variance_rffs):
    """Calculates the z_trans_z (z^T z) matrix where Z is the random
    features generated from raw input data X, for calculating
    variance only (since in this case we only use up to
    self.variance_rffs of the features generated).

    Args:
        dataset: An OnlineDataset or OfflineDataset object storing
            the data we will use.
        kernel: A valid kernel object that can generate random features.
        variance_rffs (int): The number of variance random features.

    Returns:
        z_trans_z: The cupy or numpy matrix resulting from z^T z. Will
            be shape M x M for M random features.
    """
    if kernel.device == "cpu":
        z_trans_z = np.zeros((variance_rffs, variance_rffs))
    else:
        z_trans_z = cp.zeros((variance_rffs, variance_rffs))
    if dataset.pretransformed:
        for xfeatures in dataset.get_chunked_x_data():
            z_trans_z += xfeatures[:,:variance_rffs].T @ xfeatures[:,:variance_rffs]
    else:
        for xdata in dataset.get_chunked_x_data():
            xfeatures = kernel.transform_x(xdata)
            z_trans_z += xfeatures[:,:variance_rffs].T @ xfeatures[:,:variance_rffs]
    return z_trans_z
