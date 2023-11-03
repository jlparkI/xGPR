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
        preset_ab = None):
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
        preset_ab: Either None or a two element array.
            If not None, instead of optimizing
            alpha and beta, the preset values are used and the
            score is returned. This is really only used for testing.

    Returns:
        score (float): The negative marginal log likelihood for
            the optimized alpha and beta.
        alpha_beta (ndarray): The optimized alpha and beta.
    """
    if preset_ab is not None:
        score = nll_terms[0] / preset_ab[0] + 0.5 * (ndatapoints - nrffs) * \
            np.log(preset_ab[0]) + nll_terms[1] + 0.5 * nrffs * np.log(preset_ab[1])
        return score, preset_ab

    beta = np.sqrt(2 * nll_terms[0] / (ndatapoints * lambda_**2))
    score = nll_terms[0] / (beta * lambda_)**2 + (ndatapoints - nrffs) * np.log(lambda_)
    score += nll_terms[1] + ndatapoints * np.log(beta)
    return score + 0.5 * ndatapoints * np.log(2*np.pi), beta
