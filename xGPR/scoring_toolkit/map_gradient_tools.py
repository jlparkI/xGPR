"""Contains functions for calculating MAP loss gradients
for the regression model."""
import numpy as np
try:
    import cupy as cp
except:
    pass
from ..constants import constants



def minibatch_map_gradient(params, x_data, y_data, kernel, a_reg):
    """Calculates the regularized MAP gradient using a
    minibatch supplied by caller.

    Args:
        params (np.ndarray): A numpy array of shape
            num_rffs + num hyperparams. The first num_hyperparams
            elements are the hyperparameters, the remaining
            elements are the weights.
        x_data (array): A cupy or numpy array containing the
            minibatch for which the gradient will be calculated.
        y_data (array): A cupy or numpy array containing the associated
            target (y) values.
        kernel: A kernel object that can generate random features.
        a_reg (float): A regularization parameter.

    Returns:
        loss (float): The loss value for this minibatch.
        grad (array): A cupy or numpy array (depending on device)
            containing the gradient for both hyperparameters and
            weights.
    """
    num_hparams = kernel.get_hyperparams().shape[0]

    #Unlike full map grad, input params here are either cupy or numpy
    #array, depending on device -- switching the weights back and forth
    #for each minibatch would be expensive, and we aren't using a scipy
    #routine for SGD so don't need to. Only catch: the kernel hparams are always
    #numpy array. Therefore, if running on gpu, send hparams to kernel
    #as numpy array before proceeding.
    if kernel.device == "gpu":
        grad = cp.zeros(params.shape)
        kernel.set_hyperparams(cp.asnumpy(params[:num_hparams]), logspace=True)
    else:
        grad = np.zeros(params.shape)
        kernel.set_hyperparams(params[:num_hparams], logspace=True)

    hparams = params[:num_hparams]
    weights = params[num_hparams:]

    xfeatures, dz_dsigma = kernel.kernel_specific_gradient(x_data)
    loss = map_gradient(hparams, xfeatures, y_data,
                dz_dsigma, weights, grad)
    complete_map_grad_calc(grad, hparams, x_data.shape[0], weights, a_reg)
    return loss, grad



def full_map_gradient(in_params, dataset, kernel, a_reg, verbose):
    """Calculates the regularized MAP gradient for
    the full dataset.

    Args:
        in_params (np.ndarray): A numpy array of shape
            num_rffs + num hyperparams. The first num_hyperparams
            elements are the hyperparameters, the remaining
            elements are the weights.
        dataset: A Dataset object for the full training dataset.
        kernel: A valid kernel object that can generate random
            features.
        a_reg (float): A regularization parameter.
        verbose (bool): Whether to print an update after gradient
            calculation.

    Returns:
        loss (float): The loss value calculated across the full
            dataset.
        gradient (np.ndarray): An array of shape num_rffs +
            num_hyperparams. The first num_hyperparams elements
            are the hyperparameter gradients, the remainder are
            the weight gradients.
    """
    num_hparams = kernel.get_hyperparams().shape[0]
    kernel.set_hyperparams(in_params[:num_hparams], logspace=True)

    #Input here is always numpy array (in order to be compatible with
    #Scipy's LBFGS). Since this func loops over the whole dataset, time
    #to convert from / to cupy array once at beginning is negligible
    #fraction of total expense.
    if kernel.device == "gpu":
        grad = cp.zeros(in_params.shape)
        params = cp.asarray(in_params)
        hparams, weights = cp.exp(params[:num_hparams]), params[num_hparams:]
    else:
        grad = np.zeros(in_params.shape)
        params = in_params.copy()
        hparams, weights = np.exp(params[:num_hparams]), params[num_hparams:]

    loss = 0.0
    for x_data, y_data in dataset.get_chunked_data():
        xfeatures, dz_dsigma = kernel.kernel_specific_gradient(x_data)
        loss += map_gradient(hparams, xfeatures, y_data,
                        dz_dsigma, weights, grad)
    loss += complete_map_grad_calc(grad, hparams, dataset.get_ndatapoints(),
                weights, a_reg)

    if kernel.device == "gpu":
        grad = cp.asnumpy(grad)
    if verbose:
        print("Gradient evaluation complete.")
    return loss, grad



def map_gradient(hparams, z_data, y_data, dz_dsigma,
        weights, gradient):
    """Calculates the gradient of the training set loss, given specified
    regularization parameters, for a single batch of data. The
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
    loss = 0.5 * float(weights.T @ weights) / hparams[1]**2
    loss += weights.shape[0] * log_hparams[1] + ndatapoints * log_hparams[0]
    loss += 0.5 * log_hparams[1] / (a_reg**2)
    loss += 0.5 * (log_hparams[0] - constants.LAMBDA_HPRIOR) / (a_reg**2)

    gradient[0] += ndatapoints / hparams[0]
    gradient[0] += (log_hparams[0] - constants.LAMBDA_HPRIOR) / (hparams[0] * a_reg**2)

    gradient[1] += weights.shape[0] / hparams[1]
    gradient[1] += log_hparams[1] / (hparams[1] * a_reg**2)
    gradient[1] -= float(weights.T @ weights) / hparams[1]**3

    if hparams.shape[0] > 2:
        gradient[2:hparams.shape[0]] += (log_hparams[2:] - constants.SIGMA_HPRIOR) / \
                (hparams[2:] * a_reg**2)
        loss += 0.5 * float((log_hparams[2:] - constants.SIGMA_HPRIOR).sum()) / (a_reg**2)

    gradient[hparams.shape[0]:] += weights / (hparams[1]**2)

    gradient[:hparams.shape[0]] *= hparams
    return float(loss)
