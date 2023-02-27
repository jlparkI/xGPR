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
        hyperparams (np.ndarray): This kernel has three
            hyperparameters: lambda_ (noise), beta_ (amplitude)
            and sigma (inverse mismatch tolerance).
    """

    def __init__(self, xdim, num_rffs, random_seed = 123,
                device = "cpu", num_threads = 2, double_precision = False,
                **kwargs):
        """Constructor for RBF.

        Args:
            xdim (tuple): The dimensions of the input.
            num_rffs (int): The user-requested number of random Fourier features.
            random_seed (int): The seed to the random number generator.
            device (str): One of 'cpu', 'gpu'. Indicates the starting device.
            num_threads (int): The number of threads to use if running on CPU. If
                running on GPU, this is ignored.
            double_precision (bool): If True, generate random features in double precision.
                Otherwise, generate as single precision.
        """
        super().__init__(num_rffs, xdim, num_threads,
                sine_cosine_kernel = True, random_seed = random_seed,
                double_precision = double_precision)
        self.hyperparams = np.ones((3))
        self.bounds = np.asarray([[1e-3,1e1], [0.125, 8], [1e-6, 1e2]])

        self.device = device
