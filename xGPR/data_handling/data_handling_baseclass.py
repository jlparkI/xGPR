"""This is the baseclass for OfflineDataset and OnlineDataset.

The DatasetBaseclass class contains methods shared by both
in-memory and on-disk datasets.
"""
import abc
from abc import ABC


class DatasetBaseclass(ABC):
    """DatasetBaseclass stores methods common to both
    OfflineDataset and OnlineDataset, and ensures
    that both share a common API.

    Attributes:
        xdim_ (tuple): A length 3 tuple (for 3d data) or 2
            (for 2d data) indicating the full dimensions of
            the dataset.
        chunk_size (int): Either the chunk_size for online datasets or the
            largest allowed array size for offline datasets.
        trainy_mean (float): The mean of the training y-data (for regression only).
        trainy_std (float): The standard deviation of the training y-data (for
            regression only).
        max_class (int): The largest class number (for classification only).
    """
    def __init__(self, xdim, chunk_size,
            trainy_mean, trainy_std, max_class=0):
        self._xdim = xdim

        #Trainy_mean and trainy_std are used
        #for regression, while max_class is used
        #for classification.
        self._trainy_mean = trainy_mean
        self._trainy_std = trainy_std
        self._max_class = max_class
        self._chunk_size = chunk_size


    @abc.abstractmethod
    def get_chunked_data(self):
        """Abstract method to force child class to implement
        get_chunked_data."""

    @abc.abstractmethod
    def get_chunked_x_data(self):
        """Abstract method to force child class to implement
        get_chunked_x_data."""


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
        return self._max_class + 1


    def get_xdim(self):
        """Returns the xdim list describing the size of the
        dataset."""
        return self._xdim

    def get_ndatapoints(self):
        """Returns the number of datapoints."""
        return self._xdim[0]

    def get_chunk_size(self):
        """Return the chunk size."""
        return self._chunk_size
