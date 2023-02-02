"""This module 'compresses' the output of another kernel using
SRHT operations. This can be useful if we want to project from
a high-dimensional random feature space back into a lower
more manageable space for hyperparameter tuning purposes."""
from math import ceil

import numpy as np
from cpu_basic_hadamard_operations import floatCpuSRHT, doubleCpuSRHT

try:
    import cupy as cp
    from cuda_basic_hadamard_operations import floatCudaSRHT, doubleCudaSRHT
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
        compressor_func: A reference to floatCpuSRHT, doubleCpuSRHT,
            floatCudaSRHT or doubleCudaSRHT as appropriate.
        device: Either "cpu" or "gpu".
        double_precision (bool): If True, input is assumed to be
                doubles, else floats.
        num_threads (int): Number of threads to use for CPU operations.
    """

    def __init__(self, compression_size, input_size, device = "cpu",
            double_precision = True, random_seed = 123):
        """Class constructor.

        Args:
            compression_size (int): The desired size for the output.
            input_size (int): The expected size of the input.
            device (str): The starting device.
            double_precision (bool): If True, input is assumed to be
                doubles, else floats.
            random_seed (int): A seed for the random number generator.

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
        self.num_threads = 2


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
        xfeatures = features.astype(self.dtype)
        if features.shape[1] < self.padded_dims:
            xfeatures = self.zero_arr((features.shape[0], self.padded_dims), self.dtype)
            xfeatures[:,:features.shape[1]] = features
        else:
            xfeatures = features.astype(self.dtype)
        if no_compression:
            self.compressor_func(xfeatures, self.radem, self.col_sampler,
                self.padded_dims, self.num_threads)
            return xfeatures

        self.compressor_func(xfeatures, self.radem, self.truncated_sampler,
                self.compression_size, self.num_threads)
        return xfeatures[:,:self.compression_size]


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
        """
        if value == "cpu":
            if not isinstance(self.radem, np.ndarray):
                self.radem = cp.asnumpy(self.radem)
            self.zero_arr = np.zeros
            if self.double_precision:
                self.dtype = np.float64
                self.compressor_func = doubleCpuSRHT
            else:
                self.dtype = np.float32
                self.compressor_func = floatCpuSRHT

        elif value == "gpu":
            self.radem = cp.asarray(self.radem)
            self.zero_arr = cp.zeros
            if self.double_precision:
                self.dtype = cp.float64
                self.compressor_func = doubleCudaSRHT
            else:
                self.dtype = cp.float32
                self.compressor_func = floatCudaSRHT
        else:
            raise ValueError("Unrecognized device supplied. Must be one "
                    "of 'cpu', 'gpu'.")
        self.device_ = value
