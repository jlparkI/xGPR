"""Contains functions for calculating exact and approximate
NMLL gradients for the regression model."""
import numpy as np
try:
    import cupy as cp
except:
    pass
from ..constants import constants
from .cho_solvers import gpu_cho_calcs, cpu_cho_calcs, cpu_cho_solver, gpu_cho_solver


def exact_nmll_reg_grad(z_trans_z, z_trans_y, y_trans_y,
                                hparams, ndatapoints,
                                dz_dsigma_ty,
                                inner_deriv, device):
    """Calculates the gradient for NMLL.
    The gradient calculation here is for regression. This calculation
    is for the full dataset rather than for minibatches (which can
    be calculated in a more efficient way, see next function).

    Args:
        z_trans_z: Numpy or cupy array of shape (self.num_rffs, self.num_rffs)
            containing z^T z.
        z_trans_y: Numpy or cupy array of shape (self.num_rffs) containing
            z^T y.
        y_trans_y (float): The dot product y^T y.
        hparams (np.ndarray): The hyperparameters (not the log of the hyperparameters).
        dz_dsigma_ty: A cupy or numpy array containing (dz_dsigma^T y).
            Shape is (self.num_rffs, M) where M is the number of kernel-
            specific hyperparameters.
        ndatapoints (int): The number of datapoints.
        device (str): Either "cpu" or "gpu".

    Returns:
        z_trans_z_chol: Numpy or cupy array containing the cholesky decomposition
            of (z_trans_z + lambda_ I).
        weights: Numpy or cupy array containing (z_trans_z + lambda_)^-1 z^T y.
        dnll_dlambda (float): The gradient of the NMLL w/r/t lambda_.
        dnll_dbeta (float): The gradient of the NMLL w/r/t beta_.
    """
    z_trans_z.flat[::z_trans_z.shape[0]+1] += hparams[0]**2
    if device == "gpu":
        cho_calculator, cho_solver = gpu_cho_calcs, gpu_cho_solver
    else:
        cho_calculator, cho_solver = cpu_cho_calcs, cpu_cho_solver

    weights, z_trans_z_chol, id_trace, chol_inv = cho_calculator(z_trans_z,
                    z_trans_y, hparams[0])

    grad = np.zeros((hparams.shape[0]))

    #Note that in the following, lambda_ is hparams[0] and beta_ is
    #hparams[1], both shared between all kernels.

    #First calculate gradient w/r/t lambda...
    dnll_dlambda = (1 / hparams[0]**3) * ((z_trans_y.T @ weights) - y_trans_y)
    dnll_dlambda += (1 / hparams[0]) * (weights.T @ weights)
    dnll_dlambda += (ndatapoints - z_trans_z.shape[1]) / hparams[0]
    dnll_dlambda += hparams[0] * (chol_inv**2).sum()
    grad[0] = float(dnll_dlambda)

    #All kernels have the beta hyperparameter -- calculate gradient w/r/t this...
    dnll_dbeta = (weights.T @ (z_trans_z.T @ weights)) - (z_trans_y.T @ weights)
    dnll_dbeta *= 1 / (hparams[0]**2 * hparams[1])
    dnll_dbeta += id_trace / hparams[1]
    grad[1] = float(dnll_dbeta)

    #Finally, calculate kernel-specific hyperparameter gradients.

    for i in range(grad.shape[0] - 2):
        trace_term = cho_solver(z_trans_z_chol, inner_deriv[:,:,i])
        dnll_dsigma = 2 * (weights.T @ dz_dsigma_ty[:,i])
        dnll_dsigma -= (weights.T @ (inner_deriv[:,:,i] @ weights))
        dnll_dsigma *= (-0.5 / hparams[0]**2)
        dnll_dsigma += 0.5 * trace_term.trace()
        grad[i+2] = float(dnll_dsigma)

    grad *= hparams
    return z_trans_z_chol, weights, grad




def map_gradient(hparams, z_data, y_data, dz_dsigma,
        weights, gradient):
    """Calculates the gradient of the training set loss, given specified
    regularization parameters, for the input data arrays. The
    regularization loss is not calculated in this function and
    should be added to the result by caller. Calculations are in place
    so nothing is returned.

    Args:
        hparams (np.ndarray): The set of hyperparameters at which
            to calculate the gradient.
        z_data (array): A numpy or cupy array of transformed x-data
            we will use to calculate the gradient (must be appropriate
            for the current device, caller should verify). Shape is
            N x M for M random features, N datapoints.
        y_data (array): A numpy or cupy array of y-data we will use
            to calculate the gradient (must be appropriate for the
            current device, caller should verify). Shape is N.
        dz_dsigma (array): A cupy or numpy array (as appropriate) with
            the gradient of the random features with respect to each
            kernel-specific hyperparameter. Shape is N x M x D for
            N datapoints, M random features, D kernel specific
            hyperparameters. D may be 0.
        weights (array): A cupy or numpy array (as appopriate) of the
            current weights
        device (str): One of "cpu", "gpu".
        gradient (array): A cupy or numpy array to which the
            gradient calculated here will be added.
            This enables a caller to call this function repeatedly on
            different arrays and sum the results.

    Returns:
        loss (float): The training set loss for this batch.
    """
    weight_prod = z_data @ weights
    loss = y_data - weight_prod
    gradient[0] -= 0.5 * (loss**2).sum() / hparams[0]**3
    gradient[1] -= (loss * weight_prod).sum() / (hparams[1] * hparams[0]**2)

    for i in range(dz_dsigma.shape[2]):
        weight_prod = dz_dsigma[:,:,i] @ weights
        gradient[2 + i] -= (weight_prod * loss).sum() / hparams[0]**2

    gradient[hparams.shape[0]:] += (loss[:,None] * z_data).sum(axis=0) / hparams[0]**2
    return 0.5 * float((loss**2).sum() / hparams[0]**2)


def complete_map_grad_calc(gradient, hparams, ndatapoints,
        weights, a_reg = 1.0):
    """This function computes the terms in the MAP gradient that
    are data-independent and depend only on the hyperparameters.
    Arrays are modified in place so nothing is returned.

    Args:
        gradient (array): An array of shape (num_hyperparams + num_rffs)
            containing the MAP gradient values for both hyperparameters and
            weights.
        hparams (array): An array of shape (num_hyperparams)
            containing the current hyperparameter values (not their log)
        ndatapoints (int): The number of datapoints in the minibatch
            or the dataset.
        weights (array): A cupy or numpy array containing the current weight
            values.
        a_reg (float): A regularization value that penalizes hyperparameter
            values which might cause overfitting.

    Returns:
        loss (float): The component of the loss resulting from the regularization
            terms.
    """
    log_hparams = np.log(hparams)
    loss = 0.5 * weights.shape[0] * float(weights.T @ weights) / hparams[1]**2
    loss += weights.shape[0] * log_hparams[1] + ndatapoints * log_hparams[0]
    loss += 0.5 * log_hparams[1] / (a_reg**2)
    loss += 0.5 * (log_hparams[0] + constants.LAMBDA_HPRIOR) / (a_reg**2)

    gradient[0] += ndatapoints * log_hparams[0]
    gradient[0] += (log_hparams[0] + constants.LAMBDA_HPRIOR) / (hparams[0] * a_reg**2)

    gradient[1] += weights.shape[0] / hparams[1]
    gradient[1] += log_hparams[1] / (hparams[1] * a_reg**2)
    gradient[1] -= float(weights.T @ weights) / hparams[1]**3

    if hparams.shape[0] > 2:
        gradient[2:hparams.shape[0]] += (log_hparams[2:] + constants.SIGMA_HPRIOR) / \
                (hparams[2:] * a_reg**2)
        loss += 0.5 * float((log_hparams[2:] + constants.SIGMA_HPRIOR).sum()) / (a_reg**2)

    gradient[hparams.shape[0]:] += weights / (hparams[1]**2)

    gradient[:hparams.shape[0]] *= hparams
    return float(loss)
