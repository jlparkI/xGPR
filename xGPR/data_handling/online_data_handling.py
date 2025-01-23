"""This module describes the OnlineDataset for regression and
classification.

The OnlineDataset bundles together attributes for handling
a dataset stored in memory. The dataset can return a specific
chunk of data from the list provided to it, can serve as a generator
and return all chunks in succession, or can provide a minibatch.
"""
import numpy as np

from .data_handling_baseclass import DatasetBaseclass


class OnlineDataset(DatasetBaseclass):
    """The OnlineDataset class handles datasets which are stored in
    memory. Only unique attributes not shared by parent class are
    described here.

    Attributes:
        _xdata (array): A cupy or numpy array containing the x data.
        _ydata (array): A cupy or numpy array containing the y data.
        _sequence_lengths: Either None or a cupy / numpy array containing
            the sequence lengths (for graph / sequence kernels).
    """
    def __init__(self, xdata, ydata, sequence_lengths = None,
                       chunk_size = 2000,
                       trainy_mean = 0.,
                       trainy_std = 1.,
                       max_class = 1.):
        """Constructor for the OnlineDataset class.

        Args:
            xdata (array): A numpy array containing the x data.
            ydata (array): A numpy array containing the y data.
            device (str): The current device. Must be in ["cpu", "gpu"].
            chunk_size (int): When returning chunks of data, the chunks
                are of size chunk_size. This limits memory consumption
                (by avoiding situations where we try to featurize
                too many datapoints at once).
            trainy_mean (float): The mean of the y-values. Only used for
                regression.
            trainy_std (float): The standard deviation of the y-values.
                Only used for regression.
            max_class (int): The largest category number in the data. Only
                used for classification.
        """
        super().__init__(xdata.shape, chunk_size, trainy_mean,
                trainy_std, max_class)
        self._xdata = xdata
        self._ydata = ydata
        self._sequence_lengths = sequence_lengths


    def get_chunked_data(self):
        """A generator that returns the stored data in chunks
        of size chunk_size."""
        if self._sequence_lengths is None:
            for i in range(0, self._xdim[0], self.get_chunk_size()):
                cutoff = min(i + self.get_chunk_size(), self._xdim[0])
                xchunk = self._xdata[i:cutoff,...]
                ychunk = self._ydata[i:cutoff]
                ychunk = ychunk.astype(np.float64)
                ychunk -= self._trainy_mean
                ychunk /= self._trainy_std
                yield xchunk, ychunk, None

        else:
            for i in range(0, self._xdim[0], self.get_chunk_size()):
                cutoff = min(i + self.get_chunk_size(), self._xdim[0])
                xchunk = self._xdata[i:cutoff,...]
                ychunk = self._ydata[i:cutoff]
                lchunk = self._sequence_lengths[i:cutoff]

                ychunk = ychunk.astype(np.float64)
                ychunk -= self._trainy_mean
                ychunk /= self._trainy_std
                yield xchunk, ychunk, lchunk



    def get_chunked_x_data(self):
        """A generator that loops over the xdata only in chunks
        of size chunk_size."""
        if self._sequence_lengths is None:
            for i in range(0, self._xdim[0], self.get_chunk_size()):
                cutoff = min(i + self.get_chunk_size(), self._xdim[0])
                yield self._xdata[i:cutoff,...], None
        else:
            for i in range(0, self._xdim[0], self.get_chunk_size()):
                cutoff = min(i + self.get_chunk_size(), self._xdim[0])
                xchunk = self._xdata[i:cutoff,...]
                lchunk = self._sequence_lengths[i:cutoff]
                yield xchunk, lchunk
