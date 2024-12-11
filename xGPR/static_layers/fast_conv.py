"""Provides the FastConv1d class for convolution-based feature extraction.

If a GP is equivalent to an infinitely wide 2-layer NN, we can turn it into
an infinitely wide 3-layer NN by using feature extraction then feeding
the result into the GP. The convolution-based feature extractors here
are designed to work on time series and sequences. They act like a convolutional
layer in an NN and generate a fixed-length vector so that the GP then functions
like a fully connected layer on top of a convolutional layer.
"""
import sys

import numpy as np
try:
    import cupy as cp
except:
    pass

from ..kernels.convolution_kernels.conv_feature_extractor import FHTMaxpoolConv1dFeatureExtractor


class FastConv1d:
    """Provides tools for performing convolution-based feature
    extraction on an input dataset. The input dataset should
    have x-values that are 3d arrays where shape[1] is timepoints
    or sequence locations and shape[2] is features.

    Attributes:
        conv_kernel: Kernel object of appropriate class
            (depending on the kernel type selected).
        device (str): One of ['cpu', 'cuda']. Indicates the current device
            for the FastConv1d.
        seq_width (int): The anticipated width (number of features) of
                each input sequence / series.
        num_features (int): The number of random features to generate.
            More = improved performance but slower feature extraction
            and slower model training.
    """

    def __init__(self, seq_width:int, device:str = "cpu", random_seed:int = 123,
            conv_width:int = 9, num_features:int = 512):
        """Constructor for the FastConv1d class.

        Args:
            seq_width (int): The anticipated width (number of features) of
                each input sequence / series.
            device (str): Must be one of ['cpu', 'cuda']. Indicates the current
                device for the feature extractor.
            random_seed (int): The seed to the random number generator.
                Defaults to 123.
            conv_width (list): A convolution kernel width.
            num_features (int): The number of random features to generate.
                More = improved performance but slower feature extraction
                and slower model training.

        Raises:
            ValueError: If an unrecognized kernel type or other invalid
                input is supplied.
        """
        self.seq_width = seq_width
        self.num_features = num_features

        self.conv_kernel = FHTMaxpoolConv1dFeatureExtractor(seq_width,
                            self.num_features, random_seed, device = device,
                            conv_width = conv_width)
        self.device = device


    def predict(self, x_array, sequence_lengths, chunk_size:int = 2000):
        """Performs feature extraction using a 1d convolution kernel
        and returns an array containing the result. This function should
        be used if it is desired to generate features for sequence /
        timeseries data AFTER training (i.e. when making predictions).
        Note that when training, you should use conv1d_pretrain_feat_extract,
        which takes a dataset as input rather than an array.

        Args:
            x_array: A numpy array. Should be a 3d array with same shape[1]
                and shape[2] as the training set.
            sequence_lengths (np.ndarray): A 1d numpy array with shape[0] ==
                shape[0] of x_array. Indicates the number of sequence elements
                in each datapoint of x_array (so that x_array can be zero-padded).
            chunk_size (int): The batch size in which the input data array
                will be processed. This limits memory consumption. Defaults
                to 2000.

        Returns:
            x_features: A 2d numpy array of shape (N, M) for
                N datapoints, M features that results from applying
                the feature extraction operation to the input.

        Raises:
            ValueError: If the inputs are not valid a detailed ValueError
                is raised explaining the issue.
        """
        if sequence_lengths.shape[0] != x_array.shape[0]:
            raise ValueError("The shape[0] of sequence_lengths must match the shape[0] "
                    "of x_array.")

        x_features = []
        for i in range(0, x_array.shape[0], chunk_size):
            cutoff = min(x_array.shape[0], i + chunk_size)
            if cutoff - i == 0:
                continue

            xtrans = self.conv_kernel.transform_x(x_array[i:cutoff,...],
                    sequence_lengths[i:cutoff])

            if self.device == "cuda":
                xtrans = cp.asnumpy(xtrans)

            x_features.append(xtrans)

        if self.device == "cuda":
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()

        x_features = np.vstack(x_features)
        return x_features


    @property
    def device(self):
        """Property definition for the device attribute."""
        return self.device_



    @device.setter
    def device(self, value):
        """Setter for the device attribute."""
        if value not in ["cpu", "cuda"]:
            raise ValueError("Device must be in ['cpu', 'cuda'].")

        if "cupy" not in sys.modules and value == "cuda":
            raise ValueError("You have specified the cuda fit mode but CuPy is "
                "not installed. Currently CPU only fitting is available.")

        self.conv_kernel.device = value
        self.device_ = value
