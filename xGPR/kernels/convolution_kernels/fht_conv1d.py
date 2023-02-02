"""This kernel is a fast Hadamard transform based convolutional
kernel that is the random features equivalent of the Ngram kernel."""
from math import ceil

import numpy as np
from scipy.stats import chi
try:
    import cupy as cp
    from cuda_convolution_double_hadamard_operations import doubleGpuConv1dTransform
    from cuda_convolution_float_hadamard_operations import floatGpuConv1dTransform
except:
    pass

from ..kernel_baseclass import KernelBaseclass
from cpu_convolution_double_hadamard_operations import doubleCpuConv1dTransform
from cpu_convolution_float_hadamard_operations import floatCpuConv1dTransform


class FHTConv1d(KernelBaseclass):
    """The Conv1d class is a convolutional kernel that can work
    with non-aligned time series or sequences.
    This class inherits from KernelBaseclass.
    Only attributes unique to this child are described in this docstring.

    Attributes:
        hyperparams (np.ndarray): This kernel has three
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
        radem_diag: The diagonal matrices for the SORF transform. Type is int8.
        chi_arr: A diagonal array whose elements are drawn from the chi
            distribution. Ensures the marginals of the matrix resulting
            from S H D1 H D2 H D3 are correct.
        conv_func: A reference to either CPUConv1dTransform or
            GPUConv1dTransform, both Cython functions in compiled
            code, as appropriate based on the current device.
        stride_tricks: A reference to cp.lib.stride_tricks.as_strided
            or np.lib.stride_tricks.as_strided, as appropriate based
            on the current device.
        contiguous_array: A reference to cp.ascontiguousarray or
            np.ascontiguousarray, as appropriate based on the current
            device.
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
            double_precision (bool): If True, generate random features in double precision.
                Otherwise, generate as single precision.
            kernel_spec_parms (dict): A dictionary of additional kernel-specific
                attributes. In this case, should contain 'conv_width'.

        Raises:
            ValueError: A ValueError is raised if the dimensions of the input are
                inappropriate given the conv_width.
        """
        super().__init__(num_rffs, xdim, double_precision, sine_cosine_kernel = True)
        if len(xdim) != 3:
            raise ValueError("Tried to initialize the Conv1d kernel with a 2d x-"
                    "array! x should be a 3d array for Conv1d.")
        if "conv_width" not in kernel_spec_parms:
            raise ValueError("Conv_width not supplied to conv1d kernel!")

        self.hyperparams = np.ones((3))
        rng = np.random.default_rng(random_seed)
        self.conv_width = kernel_spec_parms["conv_width"]
        if self.conv_width >= xdim[1]:
            raise ValueError("The conv_width for the convolution kernels must be "
                    "< the length of the time series / sequence.")

        self.num_slides = xdim[1] - self.conv_width + 1
        self.dim2_no_padding = self.conv_width * xdim[2]
        self.padded_dims = 2**ceil(np.log2(max(self.dim2_no_padding, 2)))
        self.init_calc_freqsize = ceil(self.num_freqs / self.padded_dims) * \
                        self.padded_dims
        self.init_calc_featsize = 2 * self.init_calc_freqsize
        self.bounds = np.asarray([[1e-3,1e1], [0.2, 5], [1e-2, 1e2]])

        radem_array = np.asarray([-1,1], dtype=np.int8)
        self.radem_diag = rng.choice(radem_array, size=(3, 1, self.init_calc_freqsize),
                                replace=True)
        self.chi_arr = chi.rvs(df=self.padded_dims, size=self.init_calc_freqsize,
                            random_state = random_seed)

        self.conv_func = None
        self.stride_tricks = None
        self.contiguous_array = None
        self.device = device



    def kernel_specific_set_device(self, new_device):
        """Called by parent class when device is changed. Moves
        some of the object parameters to the appropriate device
        and updates self.conv_func, self.stride_tricks and
        self.contiguous_array, which are convenience references
        to the numpy / cupy versions of functions required
        for generating features."""
        if new_device == "gpu":
            if self.double_precision:
                self.conv_func = doubleGpuConv1dTransform
            else:
                self.conv_func = floatGpuConv1dTransform
            self.radem_diag = cp.asarray(self.radem_diag)
            self.chi_arr = cp.asarray(self.chi_arr).astype(self.dtype)
            self.stride_tricks = cp.lib.stride_tricks.as_strided
            self.contiguous_array = cp.ascontiguousarray
        else:
            if not isinstance(self.radem_diag, np.ndarray):
                self.radem_diag = cp.asnumpy(self.radem_diag)
                self.chi_arr = cp.asnumpy(self.chi_arr).astype(self.dtype)
            else:
                self.chi_arr = self.chi_arr.astype(self.dtype)
            if self.double_precision:
                self.conv_func = doubleCpuConv1dTransform
            else:
                self.conv_func = floatCpuConv1dTransform
            self.stride_tricks = np.lib.stride_tricks.as_strided
            self.contiguous_array = np.ascontiguousarray



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
            raise ValueError("Input X must have shape[1] >= the convolution "
                    "kernel width.")
        if len(input_x.shape) != 3:
            raise ValueError("Input X must be a 3d array.")
        xtrans = self.zero_arr((input_x.shape[0], self.init_calc_featsize), self.out_type)
        reshaped_x = self.zero_arr((input_x.shape[0], self.num_slides,
                                self.padded_dims), self.dtype)
        #Stride tricks is a little dangerous. TODO: Find efficient non stride-tricks
        #way to implement this restructuring.
        x_strided = self.stride_tricks(input_x, shape=(input_x.shape[0],
                            self.num_slides, self.dim2_no_padding),
                            strides=(input_x.strides[0], input_x.shape[2] * input_x.strides[2],
                                input_x.strides[2]))
        reshaped_x[:,:,:self.dim2_no_padding] = self.contiguous_array(x_strided)
        self.conv_func(reshaped_x, self.radem_diag, xtrans, self.chi_arr, 2,
                self.hyperparams[2], mode = "conv")
        return xtrans[:,:self.num_rffs] * self.hyperparams[1]



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
            raise ValueError("Input X must have shape[1] >= the convolution "
                    "kernel width.")
        if len(input_x.shape) != 3:
            raise ValueError("Input X must be a 3d array.")
        output_x = self.zero_arr((input_x.shape[0], self.init_calc_featsize), self.out_type)
        reshaped_x = self.zero_arr((input_x.shape[0], self.num_slides,
                                self.padded_dims), self.dtype)
        x_strided = self.stride_tricks(input_x, shape=(input_x.shape[0],
                            self.num_slides, self.dim2_no_padding),
                            strides=(input_x.strides[0], input_x.shape[2] *
                                input_x.strides[2], input_x.strides[2]))
        reshaped_x[:,:,:self.dim2_no_padding] = self.contiguous_array(x_strided)
        dz_dsigma = self.conv_func(reshaped_x, self.radem_diag,
                output_x, self.chi_arr, 2, self.hyperparams[2], mode = "conv_gradient")
        output_x *= self.hyperparams[1]
        dz_dsigma *= self.hyperparams[1]

        output_x = output_x[:,:self.num_rffs]
        dz_dsigma = dz_dsigma[:,:self.num_rffs]
        return output_x, dz_dsigma.reshape((dz_dsigma.shape[0],
                            dz_dsigma.shape[1], 1))
