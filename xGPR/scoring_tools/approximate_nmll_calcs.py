"""Contains functions for calculating exact and approximate
NMLL gradients for the regression model."""
import numpy as np
from scipy.linalg import eigh_tridiagonal
try:
    import cupy as cp
except:
    pass



def estimate_logdet(alphas, betas, num_rffs, preconditioner = None):
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
            (lambda_**2 / beta_**2 + Z^T Z), where Z was calculated
            without multiplying by beta (the amplitude hyperparameter).
    """
    mat_diag = 1 / alphas
    mat_diag[1:,:] += betas[:-1,:] / alphas[:-1,:]
    upper_diag = np.sqrt(betas) / alphas
    logdets = cp.zeros((mat_diag.shape[1]))
    tr_estim = cp.zeros((mat_diag.shape[1]))

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




def estimate_nmll(dataset, kernel, logdet, x_k,
        z_trans_y, y_trans_y):
    """Estimates the NMLL using the estimated log determinant
    (estimated separately by estimate_logdet) and the other
    information acquired from SLQ.

    Args:
        dataset: A valid Dataset object.
        kernel: A valid kernel object.
        logdet (float): The estimated log determinant.
        x_k: A cupy or numpy array containing the results of
            the CG fit.
        z_trans_y: A cupy or numpy array containing the product
            Z^T y (without multiplying by the beta hyperparameter).
        y_trans_y (float): The product y^T y.

    Returns:
        nmll (float): The estimated NMLL.
    """
    ndatapoints = dataset.get_ndatapoints()
    lambda_ = kernel.get_lambda()
    beta_ = kernel.get_beta()
    num_rffs = kernel.get_num_rffs()

    nmll = y_trans_y - z_trans_y.T @ x_k[:,0]
    nmll *= 0.5 / lambda_**2
    nmll += ndatapoints * 0.5 * np.log(np.pi * 2)
    nmll += (ndatapoints - num_rffs) * np.log(lambda_)
    nmll += 0.5 * logdet
    return nmll
