"""This module 'compresses' the output of another kernel using
SRHT operations. This can be useful if we want to project from
a high-dimensional random feature space back into a lower
more manageable space for hyperparameter tuning purposes."""
from math import ceil

import numpy as np

from xGPR.xgpr_cpu_rfgen_cpp_ext import cpuRBFFeatureGen, cpuRBFGrad
from xGPR.xgpr_cpu_rfgen_cpp_ext import cpuSRHT
try:
    import cupy as cp
    from xGPR.xgpr_cuda_rfgen_cpp_ext import cudaSRHT
except:
    pass


class SRHTCompressor():
    """This class provides the tools needed to compress the output
    of another kernel to a specified dimensionality.

    Attributes:
        compression_size (int): The desired size for the output.
        input_size (int): The expected size of the input.
        padded_dims (int): The next largest power of two greater
            than input_size. Used for padding the input.
        radem: A numpy or cupy diagonal matrix drawn from a
            Rademacher distribution of shape (padded_dims).
        col_sampler (np.ndarray): A numpy array of shape (compression_size)
            that permutes the columns of the compressed data.
        compressor_func: A reference to an appropriate wrapped C++ function.
        device: Either "cpu" or "cuda".
        double_precision (bool): If True, input is assumed to be
                doubles, else floats. Right now set to True by default.
        num_threads (int): Max number of threads to use for CPU operations.
    """

    def __init__(self, compression_size, input_size, device = "cpu",
            double_precision = True, random_seed = 123, num_threads = 2):
        """Class constructor.

        Args:
            compression_size (int): The desired size for the output.
            input_size (int): The expected size of the input.
            device (str): The starting device.
            double_precision (bool): If True, input is assumed to be
                doubles, else floats.
            random_seed (int): A seed for the random number generator.
            num_threads (int): The max number of threads to use for CPU
                operations.

        Raises:
            ValueError: Raises a ValueError if the inputs are inappropriate.
        """
        self.compression_size = compression_size
        self.input_size = input_size
        if compression_size >= input_size or compression_size <= 1:
            raise ValueError("The compression size should be < "
                    "the number of rffs and > 1.")
        self.padded_dims = 2**ceil(np.log2(max(input_size, 2)))
        self.double_precision = double_precision

        radem_array = np.asarray([-1,1], dtype=np.int8)
        rng = np.random.default_rng(random_seed)
        self.radem = rng.choice(radem_array, size=(self.padded_dims),
                                replace=True)
        self.col_sampler = rng.permutation(self.padded_dims)
        self.truncated_sampler = self.col_sampler[:self.compression_size]
        self.compressor_func = None
        self.device = device
        self.num_threads = num_threads


    def transform_x(self, features, no_compression = False):
        """Performs the SRHT operation on the input and
        returns the downsized array.

        Args:
            features: A 2d cupy or numpy array (as appropriate for device)
                that will be compressed along the second dimension. Will
                be zero-padded if not already a power of 2.
            no_compression (bool): If True, do not truncate.

        Returns:
            features: A 2d cupy or numpy array containing the compressed
                feature information.
        """
        if features.shape[1] != self.input_size or len(features.shape) != 2:
            raise ValueError("Input with unexpected size passed to a compressor "
                    "module.")
        if features.shape[1] < self.padded_dims:
            xfeatures = self.zero_arr((features.shape[0], self.padded_dims), self.dtype)
            xfeatures[:,:features.shape[1]] = features
        else:
            xfeatures = features.astype(self.dtype)

        self.compressor_func(xfeatures, self.radem, self.num_threads)
        if no_compression:
            return xfeatures[:,self.col_sampler]

        return xfeatures[:,self.truncated_sampler]


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

        Note that a number of 'convenience attributes' (e.g. self.dtype,
        self.zero_arr) are set as references to either cupy or numpy functions.
        This avoids having to write two sets of functions (one for cupy, one for
        numpy) when the steps involved are the same.
        """
        if value == "cpu":
            if not isinstance(self.radem, np.ndarray):
                self.radem = cp.asnumpy(self.radem)
            self.zero_arr = np.zeros
            self.compressor_func = cpuSRHT
            if self.double_precision:
                self.dtype = np.float64
            else:
                self.dtype = np.float32

        elif value == "cuda":
            self.radem = cp.asarray(self.radem)
            self.zero_arr = cp.zeros
            self.compressor_func = cudaSRHT
            if self.double_precision:
                self.dtype = cp.float64
            else:
                self.dtype = cp.float32
        else:
            raise ValueError("Unrecognized device supplied. Must be one "
                    "of 'cpu', 'cuda'.")
        self.device_ = value
