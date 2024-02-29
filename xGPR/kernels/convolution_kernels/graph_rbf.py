"""This kernel is equivalent to the Conv1dRBF kernel but is specialized
to work on graphs, where we only ever want a convolution width of 1."""
import numpy as np
from .conv_kernel_baseclass import ConvKernelBaseclass


class GraphRBF(ConvKernelBaseclass):
    """This is similar to sequence kernels but is specialized to work
    on graphs, where the input is a sequence of node descriptions.
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
                    num_threads = 2, double_precision = False, kernel_spec_parms = {}):
        """Constructor.

        Args:
            xdim (tuple): The dimensions of the input. Either (N, D) or (N, M, D)
                where N is the number of datapoints, D is number of features
                and M is number of timepoints or sequence elements (convolution
                kernels only).
            num_rffs (int): The user-requested number of random Fourier features.
                For sine-cosine kernels (RBF, Matern), this will be saved by the
                class as num_rffs.
            random_seed (int): The seed to the random number generator.
            device (str): One of 'cpu', 'gpu'. Indicates the starting device.
            num_threads (int): The number of threads to use for random feature generation
                if running on CPU. Ignored if running on GPU.
            double_precision (bool): If True, generate random features in double precision.
                Otherwise, generate as single precision.

        Raises:
            ValueError: A ValueError is raised if the dimensions of the input are
                inappropriate given the conv_width.
        """
        super().__init__(xdim, num_rffs, random_seed,
                num_threads, double_precision, 1,
                kernel_spec_parms)

        self.hyperparams = np.ones((2))
        self.bounds = np.asarray([[1e-3,1e2], [1e-2, 1e2]])
        self.device = device
