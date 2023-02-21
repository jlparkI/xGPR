"""Contains some simple functions for generating probe
vectors for estimating NMLL."""
import numpy as np
try:
    import cupy as cp
except:
    pass

def generate_normal_probes_gpu(nsamples, num_rffs,
                    random_seed = 123,
                    preconditioner = None):
    """Generates probe vectors which may optionally be
    preconditioned (as preferred by caller) using a
    standard normal distribution.

    Args:
        nsamples (int): The number of probe vectors to generate.
        num_rffs (int): The number of random features.
        random_seed (int): A seed for the random number generator.
        preconditioner: Either None or a valid Preconditioner object.

    Returns:
        probes (cp.ndarray): An (num_rffs, nsamples) array of
            probe vectors for trace estimation.
    """
    rng = np.random.default_rng(random_seed)
    probes = rng.standard_normal(size=(num_rffs, nsamples))
    probes = cp.asarray(probes)
    if preconditioner is not None:
        probes = preconditioner.matvec_for_sampling(probes)
    return probes


def generate_radem_probes_gpu(nsamples, num_rffs,
                    random_seed = 123):
    """Generates probe vectors from a rademacher distribution.
    Args:
        nsamples (int): The number of probe vectors to generate.
        num_rffs (int): The number of random features.
        random_seed (int): A seed for the random number generator.

    Returns:
        probes (cp.ndarray): An (num_rffs, nsamples) array of
            probe vectors for trace estimation.
    """
    rng = np.random.default_rng(random_seed)
    radem = np.asarray([-1.0,1.0])
    probes = rng.choice(radem, size=(num_rffs, nsamples),
                    replace=True)
    probes /= np.linalg.norm(probes, axis=0)[None,:]
    return cp.asarray(probes)


def generate_normal_probes_cpu(nsamples, num_rffs,
                    random_seed = 123,
                    preconditioner = None):
    """Generates probe vectors which may optionally be
    preconditioned (as preferred by caller) using a
    standard normal distribution.

    Args:
        nsamples (int): The number of probe vectors to generate.
        num_rffs (int): The number of random features.
        random_seed (int): A seed for the random number generator.
        preconditioner: Either None or a valid Preconditioner object.

    Returns:
        probes (cp.ndarray): An (num_rffs, nsamples) array of
            probe vectors for trace estimation.
    """
    rng = np.random.default_rng(random_seed)
    probes = rng.standard_normal(size=(num_rffs, nsamples))
    if preconditioner is not None:
        probes = preconditioner.matvec_for_sampling(probes)
    return probes


def generate_radem_probes_cpu(nsamples, num_rffs,
                    random_seed = 123):
    """Generates probe vectors from a rademacher distribution.

    Args:
        nsamples (int): The number of probe vectors to generate.
        num_rffs (int): The number of random features.
        random_seed (int): A seed for the random number generator.

    Returns:
        probes (cp.ndarray): An (num_rffs, nsamples) array of
            probe vectors for trace estimation.
    """
    rng = np.random.default_rng(random_seed)
    radem = np.asarray([-1.0,1.0])
    probes = rng.choice(radem, size=(num_rffs, nsamples),
                        replace=True)
    probes /= np.linalg.norm(probes, axis=0)[None,:]
    return probes
