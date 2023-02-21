"""Houses two "convenience" functions for triangular solvers
for cpu and gpu that are used by KernelBaseclass in kernel_baseclass.
"""
import numpy as np
try:
    import cupy as cp
    import cupyx as cpx
except:
    pass
from scipy.linalg import solve_triangular, cho_solve



def gpu_cho_calcs(z_trans_z, z_trans_y, lambda_):
    """Calculates some key quantities (e.g. the inner derivative)
    needed for shared gradient calculations. TODO: This can be
    made more efficient, and should be rewritten / optimized
    at a lower level.

    Args:
        z_trans_z: cupy array of shape (self.num_rffs, self.num_rffs)
            containing z^T z.
        z_trans_y: cupy array of shape (self.num_rffs) containing z^T y.
        lambda_ (float): The value of the first hyperparameter (noise) shared
            between all kernels.

    Returns:
        weights: cupy array of shape (self.num_rffs) containing the
            weights.
        z_trans_z_chol: A cupy array storing the cholesky decomposition
            of z^T z. Shape is (self.num_rffs, self.num_rffs).
        id_trace (float): The trace of (Z^T Z + lambda**2)^-1 (Z^T Z).
        chol_inv: A cupy array of shape (self.num_rffs, self.num_rffs)
            storing the inverse of z_trans_z_chol.
    """
    z_trans_z_chol = cp.linalg.cholesky(z_trans_z)
    weights = gpu_cho_solver(z_trans_z_chol, z_trans_y)
    z_trans_z.flat[::z_trans_z.shape[0]+1] -= lambda_**2
    id_trace = float(gpu_cho_solver(z_trans_z_chol, z_trans_z).trace())
    chol_inv = cpx.scipy.linalg.solve_triangular(z_trans_z_chol,
                    cp.eye(z_trans_z_chol.shape[0]), lower=True)
    return weights, z_trans_z_chol, id_trace, chol_inv


def cpu_cho_calcs(z_trans_z, z_trans_y, lambda_):
    """Calculates some key quantities (e.g. the inner derivative)
    needed for shared gradient calculations. TODO: This can be
    made more efficient, and should be rewritten / optimized
    at a lower level.

    Args:
        z_trans_z: Numpy array of shape (self.num_rffs, self.num_rffs)
            containing z^T z.
        z_trans_y: Numpy array of shape (self.num_rffs) containing z^T y.
        lambda_ (float): The value of the first hyperparameter (noise) shared
            between all kernels.

    Returns:
        weights: numpy array of shape (self.num_rffs) containing the
            weights.
        z_trans_z_chol: A numpy array storing the cholesky decomposition
            of z^T z. Shape is (self.num_rffs, self.num_rffs).
        id_trace (float): The trace of (Z^T Z + lambda**2)^-1 (Z^T Z).
        chol_inv: A numpy array of shape (self.num_rffs, self.num_rffs)
            storing the inverse of z_trans_z_chol.
    """
    z_trans_z_chol = np.linalg.cholesky(z_trans_z)
    weights = cho_solve((z_trans_z_chol, True), z_trans_y)
    z_trans_z.flat[::z_trans_z.shape[0]+1] -= lambda_**2
    id_trace = float(cho_solve((z_trans_z_chol, True), z_trans_z).trace())
    chol_inv = solve_triangular(z_trans_z_chol,
                    np.eye(z_trans_z_chol.shape[0]), lower=True)
    return weights, z_trans_z_chol, id_trace, chol_inv


def cpu_cho_solver(chol_decomp, target):
    """Provides a cho_solver calc for cpu (so that
    both the cpu and gpu call share a common interface)."""
    return cho_solve((chol_decomp, True), target)

def gpu_cho_solver(chol_decomp, target):
    """Provides a cho_solver calc for gpu (so that
    both the cpu and gpu call share a common interface).
    TODO: Optimize."""
    sol = cpx.scipy.linalg.solve_triangular(chol_decomp,
                            target, lower=True)
    return cpx.scipy.linalg.solve_triangular(chol_decomp.T,
                            sol, lower=False)
