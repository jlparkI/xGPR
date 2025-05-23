"""The radial basis function -- the most generic and popular of all
kernels."""
import numpy as np
from .sorf_kernel_baseclass import SORFKernelBaseclass


class RBF(SORFKernelBaseclass):
    """The RBF is the classic kernel and a perennial favorite. A GP
    equipped with this functions as a high-dimensional smoother.
    This class inherits from SORFKernelBaseclass which in turn inherits
    from KernelBaseclass. Only attributes unique to this child are
    described in this docstring.

    Attributes:
        hyperparams (np.ndarray): This kernel has two
            hyperparameters: lambda_ (noise)
            and sigma (inverse mismatch tolerance).
    """

    def __init__(self, xdim, num_rffs, random_seed = 123,
                device = "cpu", double_precision = False,
                kernel_spec_parms = {}):
        """Constructor for RBF.

        Args:
            xdim (tuple): The dimensions of the input.
            num_rffs (int): The user-requested number of random Fourier features.
            random_seed (int): The seed to the random number generator.
            device (str): One of 'cpu', 'gpu'. Indicates the starting device.
            double_precision (bool): If True, generate random features in double precision.
                Otherwise, generate as single precision.
            kernel_spec_parms (dict): A dictionary of other kernel-specific settings.
        """
        super().__init__(num_rffs, xdim,
                sine_cosine_kernel = True, random_seed = random_seed,
                double_precision = double_precision,
                kernel_spec_parms = kernel_spec_parms)
        self.hyperparams = np.ones((2))
        self.bounds = np.asarray([[1e-3,4], [1e-6, 1e2]])

        self.device = device
