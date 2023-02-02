"""This kernel is equivalent to the FHTConv1d kernel but is specialized
to work on graphs, where we only ever want a convolution width of 1."""
from math import ceil

import numpy as np
from scipy.stats import chi
try:
    import cupy as cp
    from cuda_convolution_double_hadamard_operations import doubleGpuGraphConv1dTransform
    from cuda_convolution_float_hadamard_operations import floatGpuGraphConv1dTransform
except:
    pass

from ..kernel_baseclass import KernelBaseclass
from cpu_convolution_double_hadamard_operations import doubleCpuGraphConv1dTransform
from cpu_convolution_float_hadamard_operations import floatCpuGraphConv1dTransform


class GraphFHTConv1d(KernelBaseclass):
    """This is similar to FHTConv1d but is specialized to work
    on graphs, where the input is a sequence of node descriptions.
    This allows a few simplifications. This class inherits from KernelBaseclass.
    Only attributes unique to this child are described in this docstring.

    Attributes:
        hyperparams (np.ndarray): This kernel has three
            hyperparameters: lambda_ (noise), beta_ (amplitude)
            and sigma (inverse mismatch tolerance).
        padded_dims (int): The size of the expected input data
            after zero-padding to be a power of 2.
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

        Raises:
            ValueError: A ValueError is raised if the dimensions of the input are
                inappropriate given the conv_width.
        """
        super().__init__(num_rffs, xdim, double_precision, sine_cosine_kernel = True)
        if len(xdim) != 3:
            raise ValueError("Tried to initialize the GraphFHTConv1d kernel with a "
                    "2d x-array! x should be a 3d array for GraphFHTConv1d.")

        self.hyperparams = np.ones((3))
        self.bounds = np.asarray([[1e-3,1e1], [0.2, 5], [1e-2, 1e2]])
        rng = np.random.default_rng(random_seed)

        self.padded_dims = 2**ceil(np.log2(max(xdim[2], 2)))
        self.init_calc_freqsize = ceil(self.num_freqs / self.padded_dims) * \
                        self.padded_dims
        self.init_calc_featsize = 2 * self.init_calc_freqsize

        radem_array = np.asarray([-1,1], dtype=np.int8)

        self.radem_diag = rng.choice(radem_array, size=(3, 1, self.init_calc_freqsize),
                                replace=True)
        self.chi_arr = chi.rvs(df=self.padded_dims, size=self.init_calc_freqsize,
                            random_state = random_seed)


        self.conv_func = None
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
                self.conv_func = doubleGpuGraphConv1dTransform
            else:
                self.conv_func = floatGpuGraphConv1dTransform
            self.radem_diag = cp.asarray(self.radem_diag)
            self.chi_arr = cp.asarray(self.chi_arr).astype(self.dtype)
        else:
            if not isinstance(self.radem_diag, np.ndarray):
                self.radem_diag = cp.asnumpy(self.radem_diag)
                self.chi_arr = cp.asnumpy(self.chi_arr).astype(self.dtype)
            else:
                self.chi_arr = self.chi_arr.astype(self.dtype)
            if self.double_precision:
                self.conv_func = doubleCpuGraphConv1dTransform
            else:
                self.conv_func = floatCpuGraphConv1dTransform



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
        if len(input_x.shape) != 3:
            raise ValueError("Input X must be a 3d array.")
        xtrans = self.zero_arr((input_x.shape[0], self.init_calc_featsize), self.out_type)
        reshaped_x = self.zero_arr((input_x.shape[0], input_x.shape[1],
                                self.padded_dims), self.dtype)
        reshaped_x[:,:,:input_x.shape[2]] = input_x
        self.conv_func(reshaped_x, self.radem_diag, xtrans, self.chi_arr,
                2, self.hyperparams[2:], self.hyperparams[1])
        return xtrans[:,:self.num_rffs]



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
        if len(input_x.shape) != 3:
            raise ValueError("Input X must be a 3d array.")
        output_x = self.zero_arr((input_x.shape[0], self.init_calc_featsize), self.out_type)
        reshaped_x = self.zero_arr((input_x.shape[0], input_x.shape[1],
                                self.padded_dims), self.dtype)
        reshaped_x[:,:,:input_x.shape[2]] = input_x
        dz_dsigma = self.conv_func(reshaped_x, self.radem_diag,
                output_x, self.chi_arr, 2, self.hyperparams[2:],
                self.hyperparams[1], mode = "gradient")
        return output_x[:,:self.num_rffs], dz_dsigma[:,:self.num_rffs]
