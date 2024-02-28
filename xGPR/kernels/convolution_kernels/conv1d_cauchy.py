"""A 1d convolution kernel that is slightly more flexible than
the Conv1dRBF."""
import numpy as np
from .conv_kernel_baseclass import ConvKernelBaseclass


class Conv1dCauchy(ConvKernelBaseclass):
    """The Conv1d class is a convolutional kernel that can work
    with non-aligned time series or sequences.
    This class inherits from ConvKernelBaseclass.
    Only attributes unique to this child are described in this docstring.
    
    Attributes:
        hyperparams (np.ndarray): This kernel has two
            hyperparameters: lambda_ (noise)
            and sigma (inverse mismatch tolerance).
        bounds (np.ndarray): The builtin bounds for hyperparameter optimization,
            which can be overriden / reset by user.
    """

    def __init__(self, xdim, num_rffs, random_seed = 123, device = "cpu",
                    num_threads = 2, double_precision = False,
                    kernel_spec_parms = {}):
        """Constructor.

        Args:
            xdim (tuple): The dimensions of the input. Either (N, D) or (N, M, D)
                where N is the number of datapoints, D is number of features
                and M is number of timepoints or sequence elements (convolution
                kernels only).
            num_rffs (int): The user-requested number of random Fourier features.
                For sine-cosine kernels (RBF, Cauchy), this will be saved by the
                class as num_rffs.
            random_seed (int): The seed to the random number generator.
            device (str): One of 'cpu', 'gpu'. Indicates the starting device.
            num_threads (int): The number of threads to use for random feature generation
                if running on CPU. If running on GPU, this is ignored.
            double_precision (bool): If True, generate random features in double precision.
                Otherwise, generate as single precision.
            kernel_spec_parms (dict): A dictionary of additional kernel-specific
                attributes. In this case, should contain 'conv_width'.

        Raises:
            ValueError: A ValueError is raised if the dimensions of the input are
                inappropriate given the conv_width.
        """
        if "conv_width" not in kernel_spec_parms:
            raise ValueError("conv_width must be included as a kernel-specific "
                    "parameter if using a sequence kernel.")

        super().__init__(xdim, num_rffs, random_seed,
                num_threads, double_precision, kernel_spec_parms["conv_width"],
                kernel_spec_parms)
        rng = np.random.default_rng(random_seed)
        dstsamples = np.sqrt(rng.exponential(size=self.num_freqs))
        self.chi_arr *= dstsamples

        self.hyperparams = np.ones((2))
        self.bounds = np.asarray([[1e-3,1e1], [1e-6, 1e2]])
        self.device = device
