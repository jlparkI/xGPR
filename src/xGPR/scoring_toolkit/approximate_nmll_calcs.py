"""Contains functions for calculating exact and approximate
NMLL gradients for the regression model."""
import numpy as np
from scipy.linalg import eigh_tridiagonal
try:
    import cupy as cp
except:
    pass



def estimate_logdet(alphas, betas, num_rffs, preconditioner = None,
        device = "cpu"):
    """Estimates the log determinant term in the NMLL using the alpha
    and beta values from a conjugate gradients run against samples
    from a normal distribution with covariance matrix P (if
    preconditioning was used) or I (if no preconditioning).

    Args:
        alphas (np.ndarray): A numpy array of shape (niter, nprobes).
        betas (np.ndarray): A numpy array of shape (niter, nprobes).
        num_rffs (int): The number of RFFs.
        preconditioner: Either None or a valid preconditioner object.

    Returns:
        logdets (float): The estimated log determinant of
            (lambda_ + Z^T Z).
    """
    mat_diag = 1 / alphas
    mat_diag[1:,:] += betas[:-1,:] / alphas[:-1,:]
    upper_diag = np.sqrt(betas) / alphas
    if device == "cuda":
        logdets = cp.zeros((mat_diag.shape[1]))
        tr_estim = cp.zeros((mat_diag.shape[1]))
    else:
        logdets = np.zeros((mat_diag.shape[1]))
        tr_estim = np.zeros((mat_diag.shape[1]))


    for i in range(mat_diag.shape[1]):
        eigvals, eigvecs = eigh_tridiagonal(mat_diag[:,i],
                        upper_diag[:-1,i], lapack_driver = "stev")
        weights = eigvecs[0,:]**2
        logdets[i] += (weights * np.log(eigvals)).sum()
        tr_estim[i] += (weights / eigvals).sum()

    logdets = num_rffs * logdets.sum() / alphas.shape[1]
    if preconditioner is not None:
        logdets += preconditioner.get_logdet()
    return float(logdets)
