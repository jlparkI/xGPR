"""These are convolution kernels that can be run once on the input, and
the resulting features can be saved and then fed into another kernel,
resulting in a two-layer GP. Consequently, these can also be considered
'feature extractors', hence their names."""
from math import ceil

import numpy as np
from scipy.stats import chi
try:
    import cupy as cp
    from cuda_convolution_float_hadamard_operations import floatGpuConv1dTransform
except:
    pass

from kernel_tools import CPUConv1dMMTransform, GPUConv1dMMTransform
from cpu_convolution_double_hadamard_operations import doubleCpuConv1dTransform


class MMConv1dFeatureExtractor():
    """This class uses convolution implemented as matrix multiplication
    to provide feature generation. (Compare to the FHT Conv feature
    extractor, which uses an FHT-based procedure.)

    Attributes:
        mode (str): Indicates the type of feature generation procedure to be performed.
        seqwidth (int): The number of features per timepoint / sequence element.
        num_rffs (int): The user-requested number of random Fourier features.
        num_freqs (int): The number of filters to use. This will be the same
            as num_rffs for most modes but for some modes we generate multiple
            features per freq.
        hyperparams (np.ndarray): A length one array containing sigma
            (a distribution parameter) which is set to 1.
        conv_width (int): The width of the convolution kernel.
            This hyperparameter can be set based on experimentation
            or domain knowledge. Defaults to 9.
        filters: A stack of M x C * D numpy or cupy arrays for M
            random features, C conv width, D input features for
            performing the convolution.
        conv_func: A reference to either CPUConvMMTransform or
            GPUConvMMTransform, both Cython functions in compiled
            code, as appropriate based on the current device.
    """
    def __init__(self, seqwidth, num_rffs, random_seed = 123, device = "cpu",
                    conv_width = 9, mode = "maxpool"):
        """Class constructor.

        Args:
            seqwidth (int): The number of features per timepoint / sequence element.
            num_rffs (int): The user-requested number of random Fourier features.
            random_seed (int): The seed to the random number generator.
            device (str): One of 'cpu', 'gpu'. Indicates the starting device.
            conv_width (int): The width of the convolution kernel. Defaults to 9.
            mode (str): Indicates the type of feature extraction to be performed.

        Raises:
            ValueError: A ValueError is raised if the dimensions of the input are
                inappropriate given the conv_width.
        """
        if mode not in ["maxpool_loc", "maxpool"]:
            raise ValueError("Unrecognized mode supplied to the MM feature extractor.")
        self.mode = mode
        self.hyperparams = np.ones((1))
        rng = np.random.default_rng(random_seed)

        self.num_rffs = num_rffs
        self.num_freqs = num_rffs
        self.seqwidth = seqwidth
        self.conv_width = conv_width

        self.filters = rng.normal(size=(self.num_freqs, self.conv_width * seqwidth))

        self.conv_func = None
        self.device = device



    def kernel_specific_set_device(self, new_device):
        """Called when device is changed. Moves
        some of the object parameters to the appropriate device
        and updates self.conv_func, which is a convenience reference
        to the Cython functions used for performing
        these convolutions."""
        if new_device == "gpu":
            self.conv_func = GPUConv1dMMTransform
            self.filters = cp.asarray(self.filters).astype(cp.float32)
        else:
            if not isinstance(self.filters, np.ndarray):
                self.filters = cp.asnumpy(self.filters).astype(np.float64)
            self.conv_func = CPUConv1dMMTransform



    def transform_x(self, input_x):
        """Performs the feature generation by calling the appropriate
        Cython function, referenced as self.conv_func."""
        if input_x.shape[1] <= self.conv_width:
            raise ValueError("Input input_x must have shape[1] >= the convolution "
                    "kernel width.")
        if len(input_x.shape) != 3:
            raise ValueError("Input input_x must be a 3d array.")
        if input_x.shape[2] != self.seqwidth:
            raise ValueError("Unexpected number of features per timepoint / sequence element "
                    "on this input.")

        output_x = self.zero_arr((input_x.shape[0], self.num_rffs), self.out_type)
        input_x = input_x.astype(self.dtype)
        self.conv_func(input_x, self.filters, output_x, self.hyperparams[0],
                        mode = self.mode)
        return output_x

    @property
    def device(self):
        """Getter for the device property, which determines
        whether calculations are on CPU or GPU."""
        return self.device_

    @device.setter
    def device(self, value):
        """Setter for device, which determines whether calculations
        are on CPU or GPU. Note that each kernel must also have
        a kernel_specific_set_device function (enforced via
        an abstractmethod) to make any kernel-specific changes
        that occur when the device is switched.

        Args:
            value (str): Must be one of 'cpu', 'gpu'.

        Raises:
            ValueError: A ValueError is raised if an unrecognized
                device is passed.

        Note that a number of 'convenience attributes' (e.g. self.dtype,
        self.zero_arr) are set as references to either cupy or numpy functions.
        This avoids having to write two sets of functions (one for cupy, one for
        numpy) when the steps involved are the same.
        Also note that cupy uses float32, which is 5-10x faster on GPU; on CPU,
        float32 provides a much more modest benefit and float64 is used instead.
        """
        if value == "cpu":
            self.zero_arr = np.zeros
            self.dtype = np.float64
            self.out_type = np.float64
        elif value == "gpu":
            self.zero_arr = cp.zeros
            self.dtype = cp.float32
            self.out_type = cp.float64
        else:
            raise ValueError("Unrecognized device supplied. Must be one "
                    "of 'cpu', 'gpu'.")
        self.device_ = value
        self.kernel_specific_set_device(value)


class FHTMaxpoolConv1dFeatureExtractor():
    """A ReLU activation, global maxpool kernel that can be run
    once on the data (since it does not need any separate
    hyperparameters). It should never be used as a model kernel,
    only as a feature extractor; it is missing some attributes
    that would be required for a model kernel (e.g. gradient
    calcs).

    Attributes:
        conv_width (int): The width of the convolution kernel.
            This hyperparameter can be set based on experimentation
            or domain knowledge. Defaults to 9.
        dim2_no_padding (int): The size of the expected input data
            once reshaped for convolution, before zero padding.
        padded_dims (int): The size of the expected input data
            once reshaped for convolution, after zero-padding to
            be a power of 2.
        init_calc_featsize (int): The number of times the transform
            will need to be performed to generate the requested number
            of random features.
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
        mode (str): One of "maxpool", "maxpool_loc". Defaults to 'maxpool'.
    """

    def __init__(self, seqwidth, num_rffs, random_seed = 123, device = "cpu",
                    conv_width = 9, mode = "maxpool"):
        """Constructor for FHT_Conv1d.

        Args:
            seqwidth (int): The number of features per timepoint / sequence element.
            num_rffs (int): The user-requested number of random Fourier features.
                For sine-cosine kernels (RBF, Matern), this will be saved by the
                class as num_rffs.
            random_seed (int): The seed to the random number generator.
            device (str): One of 'cpu', 'gpu'. Indicates the starting device.
            conv_width (int): The width of the convolution kernel. Defaults to 9.
            mode (str): One of "maxpool", "maxpool_loc". Defaults to 'maxpool'.

        Raises:
            ValueError: A ValueError is raised if the dimensions of the input are
                inappropriate given the conv_width.
        """
        if mode not in ["maxpool_loc", "maxpool"]:
            raise ValueError("Unrecognized mode supplied to the FHT feature extractor.")
        rng = np.random.default_rng(random_seed)
        self.conv_width = conv_width

        self.num_rffs = num_rffs
        self.seqwidth = seqwidth
        self.dim2_no_padding = conv_width * seqwidth
        self.padded_dims = 2**ceil(np.log2(max(self.dim2_no_padding, 2)))
        self.init_calc_featsize = ceil(self.num_rffs / self.padded_dims) * \
                        self.padded_dims

        radem_array = np.asarray([-1,1], dtype=np.int8)

        self.radem_diag = rng.choice(radem_array, size=(3, 1, self.init_calc_featsize),
                                replace=True)
        self.chi_arr = chi.rvs(df=self.padded_dims, size=self.init_calc_featsize,
                            random_state = random_seed)
        self.conv_func = None
        self.stride_tricks = None
        self.contiguous_array = None
        self.device = device
        self.mode = mode


    def kernel_specific_set_device(self, new_device):
        """Called by parent class when device is changed. Moves
        some of the object parameters to the appropriate device
        and updates convenience references to numpy / cupy
        functions used for generating features."""
        if new_device == "gpu":
            self.conv_func = floatGpuConv1dTransform
            self.radem_diag = cp.asarray(self.radem_diag)
            self.chi_arr = cp.asarray(self.chi_arr).astype(cp.float32)
            self.stride_tricks = cp.lib.stride_tricks.as_strided
            self.contiguous_array = cp.ascontiguousarray
        else:
            if not isinstance(self.radem_diag, np.ndarray):
                self.radem_diag = cp.asnumpy(self.radem_diag)
                self.chi_arr = cp.asnumpy(self.chi_arr).astype(np.float64)
            self.conv_func = doubleCpuConv1dTransform
            self.stride_tricks = np.lib.stride_tricks.as_strided
            self.contiguous_array = np.ascontiguousarray


    def transform_x(self, input_x):
        """Performs the feature generation by calling the appropriate
        Cython function, referenced as self.conv_func."""
        if input_x.shape[1] <= self.conv_width:
            raise ValueError("Input input_x must have shape[1] >= the convolution "
                    "kernel width.")
        if len(input_x.shape) != 3:
            raise ValueError("Input input_x must be a 3d array.")
        if input_x.shape[2] != self.seqwidth:
            raise ValueError("Unexpected number of features per timepoint / sequence element "
                    "on this input.")

        num_slides = input_x.shape[1] - self.conv_width + 1
        output_x = self.zero_arr((input_x.shape[0], self.init_calc_featsize), self.out_type)
        reshaped_x = self.zero_arr((input_x.shape[0], num_slides,
                                self.padded_dims), self.dtype)
        #TODO: Find non stride tricks way to implement this restructing. Stride tricks
        #is a little dangerous to use.
        x_strided = self.stride_tricks(input_x, shape=(input_x.shape[0], num_slides,
                            self.dim2_no_padding),
                            strides=(input_x.strides[0], input_x.shape[2] *
                                input_x.strides[2], input_x.strides[2]))
        reshaped_x[:,:,:self.dim2_no_padding] = self.contiguous_array(x_strided)
        self.conv_func(reshaped_x, self.radem_diag,
                output_x, self.chi_arr, 2, 1.0, mode = self.mode)
        output_x = output_x[:,:self.num_rffs]
        return output_x


    @property
    def device(self):
        """Getter for the device property, which determines
        whether calculations are on CPU or GPU."""
        return self.device_

    @device.setter
    def device(self, value):
        """Setter for device, which determines whether calculations
        are on CPU or GPU. Note that each kernel must also have
        a kernel_specific_set_device function (enforced via
        an abstractmethod) to make any kernel-specific changes
        that occur when the device is switched.

        Args:
            value (str): Must be one of 'cpu', 'gpu'.

        Raises:
            ValueError: A ValueError is raised if an unrecognized
                device is passed.

        Note that a number of 'convenience attributes' (e.g. self.dtype,
        self.zero_arr) are set as references to either cupy or numpy functions.
        This avoids having to write two sets of functions (one for cupy, one for
        numpy) for each gradient calculation when the steps involved are the same.
        Also note that cupy uses float32, which is 5-10x faster on GPU; on CPU,
        float32 provides a much more modest benefit and float64 is used instead.
        """
        if value == "cpu":
            self.zero_arr = np.zeros
            self.dtype = np.float64
            self.out_type = np.float64
        elif value == "gpu":
            self.zero_arr = cp.zeros
            self.dtype = cp.float32
            self.out_type = cp.float64
        else:
            raise ValueError("Unrecognized device supplied. Must be one "
                    "of 'cpu', 'gpu'.")
        self.device_ = value
        self.kernel_specific_set_device(value)
