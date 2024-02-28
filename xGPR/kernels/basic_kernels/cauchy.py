'''The RationalQuadratic kernel is a popular alternative to the RBF.'''
import numpy as np
from .sorf_kernel_baseclass import SORFKernelBaseclass



class Cauchy(SORFKernelBaseclass):
    """The rational quadratic kernel is a popular alternative to
    the RBF. Here it is implemented with small alpha (large alpha values
    are indistinguishable from an RBF and offer no advantage). This class
    inherits from SORFKernelBaseclass which in turn inherits from
    KernelBaseclass. Only attributes unique to this child are described
    in this docstring.

    Attributes:
        hyperparams (np.ndarray): This kernel has two
            hyperparameters: lambda_ (noise)
            and sigma (inverse mismatch tolerance).
    """

    def __init__(self, xdim, num_rffs, random_seed = 123, device = "cpu",
                num_threads = 2, double_precision = False,
                kernel_spec_parms = {}):
        """Constructor.

        Args:
            xdim (tuple): The dimensions of the input.
            num_rffs (int): The user-requested number of random Fourier features.
            random_seed (int): The seed to the random number generator.
            device (str): One of 'cpu', 'gpu'. Indicates the starting device.
            num_threads (int): The number of threads to use for generating random
                features if running on CPU. If running on GPU, this is ignored.
            num_threads (int): The number of threads to use for generating random
                features if running on CPU. If running on GPU, this is ignored.
            double_precision (bool): If True, generate random features in double precision.
                Otherwise, generate as single precision.
        """
        super().__init__(num_rffs, xdim, num_threads,
                sine_cosine_kernel = True, random_seed = random_seed,
                double_precision = double_precision,
                kernel_spec_parms = kernel_spec_parms)

        rng = np.random.default_rng(random_seed)
        dstsamples = np.sqrt(rng.exponential(size=self.num_freqs))
        self.chi_arr *= dstsamples

        self.hyperparams = np.ones((2))
        self.bounds = np.asarray([[1e-3,1e1], [1e-6, 1e2]])
        self.device = device
