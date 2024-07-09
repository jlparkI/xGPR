"""Contains functions for calculating exact and approximate
NMLL gradients for the regression model."""
import numpy as np
try:
    import cupy as cp
    import cupyx as cpx
except:
    pass
from scipy.linalg import solve_triangular, cho_solve
from .alpha_beta_optimizer import optimize_alpha_beta

def calc_gradient_terms(dataset, kernel, device, subsample = 1):
    """Calculates terms needed for the gradient calculation.
    Specific for negative marginal log likelihood (NMLL),
    as opposed to loss on the training set.

    Args:
        dataset: An OnlineDataset or OfflineDataset with the
            raw data we need for these calculations.
        kernel: A valid kernel object that can generate random features.
        device (str): One of "cpu", "cuda".
        subsample (float): A value in the range [0.01,1] that indicates what
            fraction of the training set to use each time the gradient is
            calculated (the same subset is used every time). In general, 1
            will give better results, but using a subsampled subset can be
            a fast way to find the (approximate) location of a good
            hyperparameter set.

    Returns:
        z_trans_z: The M x M cupy or numpy matrix for M random features.
        z_trans_y: The length M cupy or numpy array for M random features.
        y_trans_y (float): The dot product of y with itself.
        dz_dsigma_ty (array): Derivative w/r/t kernel-specific hyperparams times y.
        inner_deriv (array): Derivative for the log determinant portion of the NMLL.
        ndatapoints (int): The number of datapoints.
    """
    if subsample > 1 or subsample < 0.01:
        raise ValueError("Subsample must be in the range [0.01, 1].")

    num_rffs = kernel.get_num_rffs()
    hparams = kernel.get_hyperparams()
    if device == "cpu":
        z_trans_z = np.zeros((num_rffs, num_rffs))
        z_trans_y = np.zeros((num_rffs))
        dz_dsigma_ty = np.zeros((num_rffs, hparams.shape[0] - 1))
        inner_deriv = np.zeros((num_rffs, num_rffs,
                                hparams.shape[0] - 1))
        transpose = np.transpose
    else:
        z_trans_z = cp.zeros((num_rffs, num_rffs))
        z_trans_y = cp.zeros((num_rffs))
        dz_dsigma_ty = cp.zeros((num_rffs, hparams.shape[0] - 1))
        inner_deriv = cp.zeros((num_rffs, num_rffs,
                                    hparams.shape[0] - 1))
        transpose = cp.transpose

    y_trans_y = 0
    ndatapoints = 0

    if subsample == 1:
        for xin, yin, ldata in dataset.get_chunked_data():
            xfeatures, dz_dsigma, ydata = kernel.gradient_x_y(xin, yin, ldata)
            z_trans_y += xfeatures.T @ ydata
            z_trans_z += xfeatures.T @ xfeatures
            y_trans_y += ydata.T @ ydata
            ndatapoints += xfeatures.shape[0]

            for i in range(dz_dsigma.shape[2]):
                dz_dsigma_ty[:,i] += dz_dsigma[:,:,i].T @ ydata
                inner_deriv[:,:,i] += dz_dsigma[:,:,i].T @ xfeatures
    else:
        rng = np.random.default_rng(123)
        for xin, yin, ldata in dataset.get_chunked_data():
            idx_size = max(1, int(subsample * xin.shape[0]))
            idx = rng.choice(xin.shape[0], idx_size, replace=False)
            xin, yin, ldata = xin[idx,...], yin[idx], ldata[idx]
            xfeatures, dz_dsigma, ydata = kernel.gradient_x_y(xin, yin, ldata)

            z_trans_y += xfeatures.T @ ydata
            z_trans_z += xfeatures.T @ xfeatures
            y_trans_y += ydata.T @ ydata
            ndatapoints += xin.shape[0]

            for i in range(dz_dsigma.shape[2]):
                dz_dsigma_ty[:,i] += dz_dsigma[:,:,i].T @ ydata
                inner_deriv[:,:,i] += dz_dsigma[:,:,i].T @ xfeatures

    inner_deriv += transpose(inner_deriv, (1,0,2))

    return z_trans_z, z_trans_y, float(y_trans_y), dz_dsigma_ty, inner_deriv, ndatapoints



def exact_nmll_reg_grad(z_trans_z, z_trans_y, y_trans_y,
                                hparams, ndatapoints,
                                dz_dsigma_ty, inner_deriv, device):
    """Calculates the gradient for NMLL. The gradient calculation here is
    for regression.

    Args:
        z_trans_z_chol: Numpy or cupy array of shape (num_rffs, num_rffs)
            containing z^T z.
        z_trans_y: Numpy or cupy array of shape (num_rffs) containing
            z^T y.
        y_trans_y (float): The dot product y^T y.
        hparams (np.ndarray): The hyperparameters (not the log of the hyperparameters).
        beta (float): The selected value for the nuisance parameter beta.
        dz_dsigma_ty: A cupy or numpy array containing (dz_dsigma^T y).
            Shape is (num_rffs, M) where M is the number of kernel-
            specific hyperparameters.
        ndatapoints (int): The number of datapoints.
        device (str): Either "cpu" or "cuda".

    Returns:
        grad (np.ndarray): A numpy array containing the gradient of the
            hyperparameters.
    """
    z_trans_z.flat[::z_trans_z.shape[0]+1] += hparams[0]**2

    if device == "cuda":
        weights, z_trans_z_chol, chol_inv = gpu_cho_calcs(z_trans_z,
                    z_trans_y, hparams[0])
        cho_solver = gpu_cho_solver
        nll1 = float(0.5 * (y_trans_y - z_trans_y.T @ weights))
        nll2 = float(cp.log(cp.diag(z_trans_z_chol)).sum())
    else:
        weights, z_trans_z_chol, chol_inv = cpu_cho_calcs(z_trans_z,
                    z_trans_y, hparams[0])
        cho_solver = cpu_cho_solver
        nll1 = float(0.5 * (y_trans_y - z_trans_y.T @ weights))
        nll2 = float(np.log(np.diag(z_trans_z_chol)).sum())

    nrffs = float(z_trans_z.shape[0])
    negloglik, beta = optimize_alpha_beta(hparams[0],
                np.array([nll1, nll2]), float(ndatapoints), nrffs)

    grad = np.zeros((hparams.shape[0]))

    #Note that in the following, lambda_ is hparams[0], beta is a
    #nuisance parameter, and alpha is lambda_ * beta.
    alpha = hparams[0] * beta

    #First calculate gradient w/r/t lambda...
    dnll_dlambda = (1 / (beta**2 * hparams[0]**3)) * ((z_trans_y.T @ weights) - y_trans_y)
    dnll_dlambda += (1 / (beta**2 * hparams[0])) * (weights.T @ weights)
    dnll_dlambda += (ndatapoints - z_trans_z_chol.shape[1]) / hparams[0]
    dnll_dlambda += hparams[0] * (chol_inv**2).sum()
    grad[0] = float(dnll_dlambda)

    #Finally, calculate kernel-specific hyperparameter gradients.

    for i in range(grad.shape[0] - 1):
        trace_term = cho_solver(z_trans_z_chol, inner_deriv[:,:,i])
        dnll_dsigma = -2 * (weights.T @ dz_dsigma_ty[:,i])
        dnll_dsigma += (weights.T @ (inner_deriv[:,:,i] @ weights))
        dnll_dsigma *= (0.5 / alpha**2)
        dnll_dsigma += 0.5 * trace_term.trace()
        grad[i+1] = float(dnll_dsigma)

    grad *= hparams
    return negloglik, grad, beta


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
    chol_inv = cpx.scipy.linalg.solve_triangular(z_trans_z_chol,
                    cp.eye(z_trans_z_chol.shape[0]), lower=True)
    return weights, z_trans_z_chol, chol_inv


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
    chol_inv = solve_triangular(z_trans_z_chol,
                    np.eye(z_trans_z_chol.shape[0]), lower=True)
    return weights, z_trans_z_chol, chol_inv
