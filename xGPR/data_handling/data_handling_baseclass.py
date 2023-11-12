"""This is the baseclass for OfflineDataset and OnlineDataset.

The DatasetBaseclass class contains methods shared by both
in-memory and on-disk datasets.
"""
import abc
from abc import ABC

import numpy as np
try:
    import cupy as cp
except:
    pass


class DatasetBaseclass(ABC):
    """DatasetBaseclass stores methods common to both
    OfflineDataset and OnlineDataset, and ensures
    that both share a common API.

    Attributes:
        device (str): Must be one of ['cpu', 'gpu']. Indicates
            the device on which calculations will be
            performed.
        xdim_ (tuple): A length 3 tuple (for 3d data) or 2
            (for 2d data) indicating the full dimensions of
            the dataset.
        mbatch_counter (int): An integer storing the file we are currently
            using to generate the next minibatch. Used when generating
            minibatches only.
        mbatch_row (int): An integer storing the location where we are
            extracting the next minibatch. Used when generating
            minibatches only.
        chunk_size (int): Either the chunk_size for online datasets or the
            largest allowed array size for offline datasets.
    """
    def __init__(self, xdim, device, chunk_size,
            trainy_mean, trainy_std, max_class):
        self.device = device
        self._xdim = xdim

        #Trainy_mean and trainy_std are used
        #for regression, while max_class is used
        #for classification.
        self._trainy_mean = trainy_mean
        self._trainy_std = trainy_std
        self._max_class = max_class

        self.mbatch_counter = 0
        self.mbatch_row = 0
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def get_chunked_data(self):
        """Abstract method to force child class to implement
        get_chunked_data."""

    @abc.abstractmethod
    def get_chunked_x_data(self):
        """Abstract method to force child class to implement
        get_chunked_x_data."""

    @abc.abstractmethod
    def get_chunked_y_data(self):
        """Abstract method to force child class to implement
        get_chunked_y_data."""


    @abc.abstractmethod
    def get_next_minibatch(self, batch_size):
        """Abstract method to force child class to
        implement get_next_minibatch."""

    def get_ymean(self):
        """Returns the mean of the training y data.
        Only useful if data is for regression."""
        return self._trainy_mean

    def get_ystd(self):
        """Returns the standard deviation of the training
        y data. Only useful if data is for regression."""
        return self._trainy_std


    def get_n_classes(self):
        """Gets the largest category number in the training
        data. Only useful if data is for classification."""
        return self._max_class


    def get_xdim(self):
        """Returns the xdim list describing the size of the
        dataset."""
        return self._xdim

    def get_ndatapoints(self):
        """Returns the number of datapoints."""
        return self._xdim[0]

    def reset_index(self):
        """Restarts minibatch collection at the beginning
        of the dataset."""
        self.mbatch_counter = 0
        self.mbatch_row = 0

    def get_chunk_size(self):
        """Return the chunk size."""
        return self.chunk_size

    @property
    def device(self):
        """Property definition for the device attribute."""
        return self.device_

    @device.setter
    def device(self, value):
        """Setter for the device attribute."""
        if value == "gpu":
            self.array_loader = cp.load
            self.dtype = cp.float64
        elif value == "cpu":
            self.array_loader = np.load
            self.dtype = np.float64
        else:
            raise ValueError("Device supplied to Dataset must be "
                        "in ['cpu', 'gpu'].")
        self.device_ = value
