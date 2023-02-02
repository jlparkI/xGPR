"""This kernel is a convolutional kernel that is the random features
equivalent of the Ngram kernel."""
import numpy as np
try:
    import cupy as cp
except:
    pass

from kernel_tools import CPUConv1dMMTransform, GPUConv1dMMTransform
from ..kernel_baseclass import KernelBaseclass


class MMConv1d(KernelBaseclass):
    """The MMConv1d class is a convolutional kernel that can work
    with non-aligned time series or sequences.
    This class inherits from KernelBaseclass. Only attributes
    unique to this child are described in this docstring.

    Attributes:
        hyperparams (np.ndarray): A length three array of three
            hyperparameters: lambda_ (noise), beta_ (amplitude)
            and sigma (inverse mismatch tolerance).
        conv_width (int): The width of the convolution kernel.
            This hyperparameter can be set based on experimentation
            or domain knowledge. Defaults to 9.
        num_slides (int): The length of the (zero-padded) input data
            minus conv_width + 1. The number of points that will
            result from the convolution.
        dim2_no_padding (int): The size of the expected input data
            once reshaped for convolution, before zero padding.
        padded_dims (int): The size of the expected input data
            once reshaped for convolution, after zero-padding to
            be a power of 2.
        init_calc_freqsize (int): The number of times the transform
            will need to be performed to generate the requested number
            of sampled frequencies.
        init_calc_featsize (int): The number of features generated initially
            (before discarding excess).
        conv_func: A reference to either CPUConvMMTransform or
            GPUConvMMTransform, both Cython functions in compiled
            code, as appropriate based on the current device.
        stride_tricks: A reference to cp.lib.stride_tricks.as_strided
            or np.lib.stride_tricks.as_strided, as appropriate based
            on the current device.
    """

    def __init__(self, xdim, num_rffs, random_seed = 123, device = "cpu",
                    double_precision = True, kernel_spec_parms = {}):
        """Constructor for FHT_Conv1d.

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
            conv_width (int): The width of the convolution kernel. Defaults to 9.
            double_precision (bool): Not used for this kernel; accepted to preserve
                common interface with other kernels.
            kernel_spec_parms (dict): A dictionary of additional kernel-specific
                attributes. In this case, should contain 'conv_width'.

        Raises:
            ValueError: A ValueError is raised if the dimensions of the input are
                inappropriate given the conv_width.
        """
        super().__init__(num_rffs, xdim, double_precision, sine_cosine_kernel = True)
        if "conv_width" not in kernel_spec_parms:
            raise ValueError("For a convolution kernel, 'conv_width' must be supplied.")
        if len(xdim) != 3:
            raise ValueError("Tried to initialize the Conv1d kernel with a 2d x-"
                    "array! x should be a 3d array for Conv1d.")
        self.hyperparams = np.ones((3))
        rng = np.random.default_rng(random_seed)
        self.conv_width = kernel_spec_parms["conv_width"]
        if self.conv_width >= xdim[1]:
            raise ValueError("The conv_width for the convolution kernels must be "
                    "< the length of the time series / sequence.")

        self.num_slides = xdim[1] - self.conv_width + 1
        self.bounds = np.asarray([[1e-3,1e1], [0.2, 5], [1e-6, 1e2]])

        self.filters = rng.normal(size=(self.num_freqs, self.conv_width * xdim[2]))

        self.conv_func = None
        self.stride_tricks = None
        self.dtype = None
        self.device = device


    def kernel_specific_set_device(self, new_device):
        """Called when device is changed. Moves
        some of the object parameters to the appropriate device
        and updates self.conv_func, which is a convenience reference
        to the Cython functions used for performing
        these convolutions."""
        if new_device == "gpu":
            self.dtype = cp.float32
            self.conv_func = GPUConv1dMMTransform
            self.filters = cp.asarray(self.filters).astype(cp.float32)
        else:
            self.dtype = np.float64
            if not isinstance(self.filters, np.ndarray):
                self.filters = cp.asnumpy(self.filters).astype(np.float64)
            self.conv_func = CPUConv1dMMTransform


    def transform_x(self, input_x):
        """Generates random features.

        Args:
            input_x: A numpy or cupy array containing the raw input data.

        Returns:
            xtrans: A cupy or numpy array containing the generated features.

        Raises:
            ValueError: A value error is raised if the dimensionality of the
                input does not meet validity criteria.
        """
        if input_x.shape[1] <= self.conv_width:
            raise ValueError("Input input_x must have shape[1] >= the convolution "
                    "kernel width.")
        if len(input_x.shape) != 3:
            raise ValueError("Input input_x must be a 3d array.")

        input_x = input_x.astype(self.dtype)
        output_x = self.zero_arr((input_x.shape[0], self.num_rffs), self.out_type)
        self.conv_func(input_x, self.filters, output_x, self.hyperparams[2],
                        mode = "conv")
        return self.hyperparams[1] * output_x



    def kernel_specific_gradient(self, input_x):
        """Since all kernels share the beta and lambda hyperparameters,
        the gradient for these can be calculated by the parent class.
        The gradient kernel-specific hyperparameters however is calculated
        using an array (dz_dsigma) specific to each
        kernel. The kernel-specific arrays are calculated here.

        Args:
            input_x: A cupy or numpy array containing the raw input data.

        Returns:
            output_x: A cupy or numpy array containing the random feature
                representation of the input.
            dz_dsigma: A cupy or numpy array containing the derivative of
                output_x with respect to the kernel-specific hyperparameters.
        """
        if input_x.shape[1] <= self.conv_width:
            raise ValueError("Input input_x must have shape[1] >= the convolution "
                    "kernel width.")
        if len(input_x.shape) != 3:
            raise ValueError("Input input_x must be a 3d array.")

        output_x = self.zero_arr((input_x.shape[0], self.num_rffs), self.out_type)
        input_x = input_x.astype(self.dtype)
        gradient = self.conv_func(input_x, self.filters, output_x, self.hyperparams[2],
                        mode = "conv_grad")
        gradient = gradient.reshape((gradient.shape[0], gradient.shape[1], 1))
        gradient *= self.hyperparams[1]
        output_x *= self.hyperparams[1]
        return output_x, gradient
