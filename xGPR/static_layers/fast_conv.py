"""Provides the FastConv1d class for convolution-based feature extraction.

If a GP is equivalent to an infinitely wide 2-layer NN, we can turn it into
an infinitely wide 3-layer NN by using feature extraction then feeding
the result into the GP. The convolution-based feature extractors here
are designed to work on time series and sequences. They act like a convolutional
layer in an NN and generate a fixed-length vector so that the GP then functions
like a fully connected layer on top of a convolutional layer.
"""
import sys
import os

import numpy as np
try:
    import cupy as cp
except:
    pass

from ..kernels.convolution_kernels.conv_feature_extractor import FHTMaxpoolConv1dFeatureExtractor
from ..kernels.convolution_kernels.conv_feature_extractor import MMConv1dFeatureExtractor
from ..data_handling.offline_data_handling import OfflineDataset

class FastConv1d:
    """Provides tools for performing convolution-based feature
    extraction on an input dataset. The input dataset should
    have x-values that are 3d arrays where shape[1] is timepoints
    or sequence locations and shape[2] is features.

    Attributes:
        conv_kernel (list): List of kernel objects of appropriate classes
            (depending on the kernel type selected).
        device (str): One of ['cpu', 'gpu']. Indicates the current device
            for the FastConv1d.
        seq_width (int): The anticipated width (number of features) of
                each input sequence / series.
        num_features (int): The number of random features to generate.
            More = improved performance but slower feature extraction
            and slower model training.
        f_per_kernel (int): The number of features per convolution kernel
            width.
        zero_arr: A convenience reference to either np.zeros or cp.zeros,
            depending on device.
    """

    def __init__(self, seq_width, device = "cpu", random_seed = 123,
            conv_width = [9], num_features = 512, mode = "maxpool"):
        """Constructor for the FastConv1d class.

        Args:
            seq_width (int): The anticipated width (number of features) of
                each input sequence / series.
            device (str): Must be one of ['cpu', 'gpu']. Indicates the current
                device for the feature extractor.
            random_seed (int): The seed to the random number generator.
                Defaults to 123.
            conv_width (list): A list of convolution kernel widths,
                up to three in length. If the length of the list is
                > 1, the num_features must be an integer multiple of
                the length of the list. num_features / len(conv_width)
                features are generated for each convolution kernel
                width.
            num_features (int): The number of random features to generate.
                More = improved performance but slower feature extraction
                and slower model training. Defaults to 512.
            mode (str): One of 'maxpool', 'maxpool_loc'.
                'maxpool_loc' adjusts the result based on global mean pooling.

        Raises:
            ValueError: If an unrecognized kernel type or other invalid
                input is supplied.
        """
        self.zero_arr = np.zeros
        if len(conv_width) > 3:
            raise ValueError("Currently only up to three conv_widths "
                    "are generated at one time.")
        if num_features % len(conv_width) != 0:
            raise ValueError("The number of features for the FastConv1d "
                    "must be an integer multiple of the number of "
                    "desired convolution widths.")

        self.f_per_kernel = int(num_features / len(conv_width))
        self.conv_kernel = [FHTMaxpoolConv1dFeatureExtractor(seq_width,
                            self.f_per_kernel, random_seed, device = device,
                            conv_width = c, mode = mode) for c in conv_width]
        self.device = device

        self.seq_width = seq_width
        self.num_features = num_features



    def conv1d_pretrain_feat_extract(self, input_dataset, output_dir):
        """Performs feature extraction using a 1d convolution kernel,
        saves the results to a specified location, and returns an
        OfflineDataset. This function should be used if it is
        desired to generate features for sequence / timeseries data
        prior to training. By use of this feature, the GP is essentially
        turned into a three-layer network with a convolutional layer
        followed by a fully-connected layer. Note that when
        making predictions, you should use conv1d_x_feat_extract,
        which takes an x-array as input rather than a dataset.

        Args:
            input_dataset: Object of class OnlineDataset or OfflineDataset.
                You should generate this object using either the
                build_online_dataset, build_offline_fixed_vector_dataset
                or build_offline_sequence_dataset functions under
                data_handling.dataset_builder.
            output_dir (str): A valid directory filepath where the output
                can be saved.

        Returns:
            conv1d_dataset (OfflineDataset): An OfflineDataset containing
                the xfiles and yfiles that resulted from the feature
                extraction operation. You can feed this directly to
                the hyperparameter tuning and fitting methods.

        Raises:
            ValueError: If the inputs are not valid a detailed ValueError
                is raised explaining the issue.
        """
        start_dir = os.getcwd()
        try:
            os.chdir(output_dir)
        except:
            raise ValueError("Invalid output directory supplied to the "
                    "feature extractor.")


        input_dataset.device = self.device
        xfiles, yfiles = [], []
        fnum = 0
        for xbatch, ybatch in input_dataset.get_chunked_data():
            xfile, yfile = f"CONV1d_FEATURES_{fnum}_X.npy", f"CONV1d_FEATURES_{fnum}_Y.npy"
            xtrans = self.zero_arr((xbatch.shape[0], self.num_features))
            for i, kernel in enumerate(self.conv_kernel):
                start, end = i * self.f_per_kernel, (i+1) * self.f_per_kernel
                xtrans[:,start:end] = kernel.transform_x(xbatch)
            if self.device == "gpu":
                ybatch = cp.asnumpy(ybatch)
                xtrans = cp.asnumpy(xtrans)
            np.save(xfile, xtrans)
            np.save(yfile, ybatch)
            xfiles.append(xfile)
            yfiles.append(yfile)
            fnum += 1

        xdim = (input_dataset.get_ndatapoints(), self.num_features)
        updated_dataset = OfflineDataset(xfiles, yfiles,
                            xdim, input_dataset.get_ymean(),
                            input_dataset.get_ystd())

        os.chdir(start_dir)
        return updated_dataset


    def conv1d_x_feat_extract(self, x_array, chunk_size = 2000):
        """Performs feature extraction using a 1d convolution kernel
        and returns an array containing the result. This function should
        be used if it is desired to generate features for sequence /
        timeseries data AFTER training (i.e. when making predictions).
        Note that when training, you should use conv1d_pretrain_feat_extract,
        which takes a dataset as input rather than an array.

        Args:
            x_array: A numpy array. Should be a 3d array with same shape[1]
                and shape[2] as the training set.
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

        x_features = []
        for i in range(0, x_array.shape[0], chunk_size):
            cutoff = min(x_array.shape[0], i + chunk_size)
            xtrans = self.zero_arr((cutoff - i, self.num_features))
            if xtrans.shape[0] == 0:
                continue
            for j, kernel in enumerate(self.conv_kernel):
                start, end = j * self.f_per_kernel, (j+1) * self.f_per_kernel
                if self.device == "gpu":
                    x_in = cp.asarray(x_array[i:cutoff,:,:]).astype(cp.float32)
                else:
                    x_in = x_array[i:cutoff,:,:]
                xtrans[:,start:end] = kernel.transform_x(x_in)

            if self.device == "gpu":
                xtrans = cp.asnumpy(xtrans).astype(np.float64)
            x_features.append(xtrans)

        x_features = np.vstack(x_features)
        return x_features


    @property
    def device(self):
        """Property definition for the device attribute."""
        return self.device_



    @device.setter
    def device(self, value):
        """Setter for the device attribute."""
        if value not in ["cpu", "gpu"]:
            raise ValueError("Device must be in ['cpu', 'gpu'].")

        if "cupy" not in sys.modules and value == "gpu":
            raise ValueError("You have specified the gpu fit mode but CuPy is "
                "not installed. Currently CPU only fitting is available.")

        if value == "cpu":
            self.zero_arr = np.zeros
        else:
            self.zero_arr = cp.zeros
        for kernel in self.conv_kernel:
            kernel.device = value
        self.device_ = value
