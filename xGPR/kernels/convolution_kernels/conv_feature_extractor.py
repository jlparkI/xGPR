"""These are convolution kernels that can be run once on the input, and
the resulting features can be saved and then fed into another kernel,
resulting in a two-layer GP. Consequently, these can also be considered
'feature extractors', hence their names."""
from math import ceil

import numpy as np
from scipy.stats import chi

from xGPR.xgpr_cpu_rfgen_cpp_ext import cpuConv1dMaxpool
try:
    import cupy as cp
    from xGPR.xgpr_cuda_rfgen_cpp_ext import cudaConv1dMaxpool
except:
    pass



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
        radem_diag: The diagonal matrices for the SORF transform. Type is int8.
        chi_arr: A diagonal array whose elements are drawn from the chi
            distribution. Ensures the marginals of the matrix resulting
            from S H D1 H D2 H D3 are correct.
        num_threads (int): Number of threads to use if running on CPU;
            ignored if running on GPU.
        simplex_rffs (bool): If True, use the simplex modification (Reid et al. 2023).
    """

    def __init__(self, seqwidth, num_rffs, random_seed = 123, device = "cpu",
                    conv_width = 9, num_threads = 2, simplex_rffs = False):
        """Constructor for FHT_Conv1d.

        Args:
            seqwidth (int): The number of features per timepoint / sequence element.
            num_rffs (int): The user-requested number of random Fourier features.
                For sine-cosine kernels (RBF, Matern), this will be saved by the
                class as num_rffs.
            random_seed (int): The seed to the random number generator.
            device (str): One of 'cpu', 'cuda'. Indicates the starting device.
            conv_width (int): The width of the convolution kernel. Defaults to 9.
            num_threads (int): Number of threads to use if running on CPU;
                ignored if running on GPU.
            simplex_rffs (bool): If True, use the simplex modification (Reid et al. 2023).

        Raises:
            ValueError: A ValueError is raised if the dimensions of the input are
                inappropriate given the conv_width.
        """
        self.simplex_rffs = simplex_rffs

        rng = np.random.default_rng(random_seed)
        self.conv_width = conv_width

        self.num_rffs = num_rffs
        self.seqwidth = seqwidth
        self.dim2_no_padding = conv_width * seqwidth
        self.padded_dims = 2**ceil(np.log2(max(self.dim2_no_padding, 2)))
        init_calc_featsize = ceil(self.num_rffs / self.padded_dims) * \
                        self.padded_dims

        radem_array = np.asarray([-1,1], dtype=np.int8)

        self.radem_diag = rng.choice(radem_array, size=(3, 1, init_calc_featsize),
                                replace=True)
        self.chi_arr = chi.rvs(df=self.padded_dims, size=self.num_rffs,
                            random_state = random_seed).astype(np.float32)

        self.num_threads = num_threads
        self.device = device


    def kernel_specific_set_device(self, new_device):
        """Called by parent class when device is changed. Moves
        some of the object parameters to the appropriate device
        and updates convenience references to numpy / cupy
        functions used for generating features."""
        if new_device == "cuda":
            self.radem_diag = cp.asarray(self.radem_diag)
            self.chi_arr = cp.asarray(self.chi_arr)
        elif not isinstance(self.radem_diag, np.ndarray):
            self.radem_diag = cp.asnumpy(self.radem_diag)
            self.chi_arr = cp.asnumpy(self.chi_arr)


    def transform_x(self, input_x, sequence_length):
        """Performs the feature generation by calling the appropriate
        Cython function, referenced as self.conv_func."""
        if sequence_length is None:
            raise ValueError("sequence_length is required for convolution kernels.")
        if input_x.shape[2] != self.seqwidth:
            raise ValueError("Unexpected number of features per timepoint / sequence element "
                    "on this input.")

        sequence_length = sequence_length.astype(np.int32, copy=False)

        if self.device == "cpu":
            output_x = np.zeros((input_x.shape[0], self.num_rffs), np.float64)
            x_in = input_x.astype(np.float32, copy=False)
            cpuConv1dMaxpool(x_in, output_x, self.radem_diag, self.chi_arr,
                    sequence_length, self.conv_width, self.num_threads, self.simplex_rffs)
        else:
            output_x = cp.zeros((input_x.shape[0], self.num_rffs), cp.float64)
            x_in = cp.asarray(input_x).astype(cp.float32, copy=False)
            cudaConv1dMaxpool(x_in, output_x, self.radem_diag, self.chi_arr,
                    sequence_length, self.conv_width, self.simplex_rffs)

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
            value (str): Must be one of 'cpu', 'cuda'.

        Raises:
            ValueError: A ValueError is raised if an unrecognized
                device is passed.
        """
        if value not in ('cuda', 'cpu'):
            raise ValueError("Unrecognized device supplied. Must be one "
                    "of 'cpu', 'cuda'.")
        self.device_ = value
        self.kernel_specific_set_device(value)
