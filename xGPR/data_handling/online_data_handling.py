"""This module describes the OnlineDataset for regression and
classification.

The OnlineDataset bundles together attributes for handling
a dataset stored in memory. The dataset can return a specific
chunk of data from the list provided to it, can serve as a generator
and return all chunks in succession, or can provide a minibatch.
"""
try:
    import cupy as cp
except:
    pass

from .data_handling_baseclass import DatasetBaseclass


class OnlineDataset(DatasetBaseclass):
    """The OnlineDataset class handles datasets which are stored in
    memory. Only unique attributes not shared by parent class are
    described here.

    Attributes:
        _xdata (array): A cupy or numpy array containing the x data.
        _ydata (array): A cupy or numpy array containing the y data.
    """
    def __init__(self, xdata, ydata,
                       device = "cpu",
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
        super().__init__(xdata.shape, device, chunk_size, trainy_mean,
                trainy_std, max_class)
        self._xdata = xdata
        self._ydata = ydata


    def get_chunked_data(self):
        """A generator that returns the stored data in chunks
        of size chunk_size."""
        for i in range(0, self._xdim[0], self.chunk_size):
            cutoff = min(i + self.chunk_size, self._xdim[0])
            xchunk = self._xdata[i:cutoff,...]
            ychunk = self._ydata[i:cutoff]
            if self.device == "gpu":
                xchunk = cp.asarray(xchunk)
                ychunk = cp.asarray(ychunk)
            ychunk = ychunk.astype(self.dtype)
            ychunk -= self._trainy_mean
            ychunk /= self._trainy_std
            yield xchunk, ychunk


    def get_chunked_x_data(self):
        """A generator that loops over the xdata only in chunks
        of size chunk_size."""
        for i in range(0, self._xdim[0], self.chunk_size):
            cutoff = min(i + self.chunk_size, self._xdim[0])
            xchunk = self._xdata[i:cutoff,...]
            if self.device == "gpu":
                xchunk = cp.asarray(xchunk)
            yield xchunk

    def get_chunked_y_data(self):
        """A generator that loops over the ydata only in chunks
        of size chunk_size."""
        for i in range(0, self._xdim[0], self.chunk_size):
            cutoff = min(i + self.chunk_size, self._xdim[0])
            ychunk = self._ydata[i:cutoff]
            if self.device == "gpu":
                ychunk = cp.asarray(ychunk)
            ychunk = ychunk.astype(self.dtype)
            ychunk -= self._trainy_mean
            ychunk /= self._trainy_std
            yield ychunk


    def get_next_minibatch(self, batch_size):
        """Gets the next minibatch (for stochastic gradient descent).

        Args:
            batch_size (int): The size of the desired minibatch.

        Returns:
            xout (array): A numpy or cupy array of x data.
            yout (array): A numpy or cupy array of y data.
            end_epoch (bool): If True, we are at the end of an epoch.
        """
        end_epoch = False
        if self.mbatch_counter >= self._xdata.shape[0]:
            end_epoch = True
            self.mbatch_counter = 0
        ssize = min(self.mbatch_counter + batch_size, self._xdata.shape[0])

        if len(self._xdim) == 3:
            xchunk = self._xdata[self.mbatch_counter:ssize,:,:]
        else:
            xchunk = self._xdata[self.mbatch_counter:ssize,:]

        ychunk = self._ydata[self.mbatch_counter:ssize]
        self.mbatch_counter += batch_size
        if self.device == "gpu":
            xchunk = cp.asarray(xchunk)
            ychunk = cp.asarray(ychunk)

        ychunk = ychunk.astype(self.dtype)
        ychunk -= self._trainy_mean
        ychunk /= self._trainy_std
        return xchunk, ychunk, end_epoch

    def get_xdata(self):
        """Returns all xdata as a single array."""
        return self._xdata

    def get_ydata(self):
        """Returns all ydata as a single array."""
        return self._ydata
