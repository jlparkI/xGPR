"""The noise hyperparameter (lambda) is actually the
ratio of two hyperparameters: alpha / beta. The choice
of alpha and beta does not however affect fitting or
variance as long as the ratio is fixed; it only
affects the marginal likelihood. This module
provides tools for optimizing the values of alpha
and beta for a given lambda so that the marginal
likelihood reported for a given lambda is as
favorable as possible."""
import numpy as np


def optimize_alpha_beta(lambda_, nll_terms, ndatapoints, nrffs,
        beta_max = 10., beta_min = 0.1):
    """Optimizes alpha and beta for a given value
    of lambda_ so that the marginal likelihood is
    at its best value for that setting for lambda_.

    Args:
        lambda_ (float): The ratio of alpha / beta.
        nll_terms (ndarray): A numpy array with two elements:
            -0.5 * y^T Zw + 0.5 * y^T y and 0.5 * ln|Z^T Z + lambda_ I|.
            These are used to calculate the marginal likelihood
            for different values of alpha and beta.
        ndatapoints (int): The number of datapoints.
        nrffs (int): The number of random features or features.
        beta_max (float): The max value for beta.
        beta_min (float): The min value for beta.

    Returns:
        score (float): The negative marginal log likelihood for
            the optimized alpha and beta.
        alpha_beta (ndarray): The optimized alpha and beta.
    """
    beta = np.sqrt(2 * nll_terms[0] / (ndatapoints * lambda_**2))
    beta = max(min(beta, beta_max), beta_min)
    score = nll_terms[0] / (beta * lambda_)**2 + (ndatapoints - nrffs) * np.log(lambda_)
    score += nll_terms[1] + ndatapoints * np.log(beta)
    return score + 0.5 * ndatapoints * np.log(2*np.pi), beta
