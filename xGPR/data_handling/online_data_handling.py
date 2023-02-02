"""This module describes the OnlineDataset class.

The OnlineDataset class bundles together attributes for handling
a dataset stored in memory. These objects are generally created by
the build_dataset method of GPModelBaseclass and should not be
created directly by the user. The dataset can return a specific
chunk of data from the list provided to it, can serve as a generator
and return all chunks in succession, or can provide a minibatch.
"""
import numpy as np
try:
    import cupy as cp
except:
    pass

from .data_handling_baseclass import DatasetBaseclass


class OnlineDataset(DatasetBaseclass):
    """The OnlineDataset class handles datasets which are stored in
    memory. It should be created using the build_dataset method of
    GPModelBaseclass, which performs a number of checks to ensure
    the data is valid before creating this object. Creating the object
    directly without using build_dataset bypasses those checks and is
    not recommended.

    Attributes:
        xdata_ (array): A cupy or numpy array containing the x data.
        trainy_mean (float): The mean of the y-values.
        trainy_std (float): The standard deviation of the y-values.
        ydata_ (array): A cupy or numpy array containing the y data.
        xdim_ (tuple): A list of length 2 (for 2d arrays) or 3 (for 3d
            arrays) where element 0 is the number of datapoints in the
            dataset, element 1 is shape[1] and element 2 is shape[2]
            (for 3d arrays for convolution kernels only).
        device (str): The current device. Must be in ["cpu", "gpu"].
        chunk_size (int): When returning chunks of data, the chunks
            are of size chunk_size. This limits memory consumption
            (by avoiding situations where we try to featurize
            too many datapoints at once).
        pretransformed_ (bool): If True, the data has already been
            "featurized" or transformed and the xfiles do not
            need to be run through the kernel. This can save time
            during fitting if an SSD is available. Default is False.
    """
    def __init__(self, xdata, ydata,
                       device = "cpu",
                       chunk_size = 2000,
                       pretransformed = False):
        """Constructor for the OnlineDataset class.

        Args:
            xdata (array): A numpy array containing the x data.
            ydata (array): A numpy array containing the y data.
            device (str): The current device. Must be in ["cpu", "gpu"].
            chunk_size (int): When returning chunks of data, the chunks
                are of size chunk_size. This limits memory consumption
                (by avoiding situations where we try to featurize
                too many datapoints at once).
            pretransformed_ (bool): If True, the data has already been
                "featurized" or transformed and the xfiles do not
                need to be run through the kernel. This can save time
                during fitting if an SSD is available. Default is False.
        """
        super().__init__(pretransformed, xdata.shape, device, chunk_size)
        if pretransformed:
            raise ValueError("Only offline datasets can be pretransformed.")
        self.xdata_ = xdata
        self.ydata_ = ydata
        self.trainy_mean_ = np.mean(ydata)
        self.trainy_std_ = np.std(ydata)



    def get_chunked_data(self):
        """A generator that returns the stored data in chunks
        of size chunk_size."""
        for i in range(0, self.xdim_[0], self.chunk_size):
            cutoff = min(i + self.chunk_size, self.xdim_[0])
            xchunk = self.xdata_[i:cutoff,...]
            ychunk = self.ydata_[i:cutoff]
            if self.device == "gpu":
                xchunk = cp.asarray(xchunk)
                ychunk = cp.asarray(ychunk)
            ychunk = ychunk.astype(self.dtype)
            ychunk -= self.trainy_mean_
            ychunk /= self.trainy_std_
            yield xchunk, ychunk


    def get_chunked_x_data(self):
        """A generator that loops over the xdata only in chunks
        of size chunk_size."""
        for i in range(0, self.xdim_[0], self.chunk_size):
            cutoff = min(i + self.chunk_size, self.xdim_[0])
            xchunk = self.xdata_[i:cutoff,...]
            if self.device == "gpu":
                xchunk = cp.asarray(xchunk)
            yield xchunk

    def get_chunked_y_data(self):
        """A generator that loops over the ydata only in chunks
        of size chunk_size."""
        for i in range(0, self.xdim_[0], self.chunk_size):
            cutoff = min(i + self.chunk_size, self.xdim_[0])
            ychunk = self.ydata_[i:cutoff]
            if self.device == "gpu":
                ychunk = cp.asarray(ychunk)
            ychunk = ychunk.astype(self.dtype)
            ychunk -= self.trainy_mean_
            ychunk /= self.trainy_std_
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
        if self.mbatch_counter >= self.xdata_.shape[0]:
            end_epoch = True
            self.mbatch_counter = 0
        ssize = min(self.mbatch_counter + batch_size, self.xdata_.shape[0])

        if len(self.xdim_) == 3:
            xchunk = self.xdata_[self.mbatch_counter:ssize,:,:]
        else:
            xchunk = self.xdata_[self.mbatch_counter:ssize,:]

        ychunk = self.ydata_[self.mbatch_counter:ssize]
        self.mbatch_counter += batch_size
        if self.device == "gpu":
            xchunk = cp.asarray(xchunk)
            ychunk = cp.asarray(ychunk)

        ychunk = ychunk.astype(self.dtype)
        ychunk -= self.trainy_mean_
        ychunk /= self.trainy_std_
        return xchunk, ychunk, end_epoch

    def get_ymean(self):
        """Returns the mean of the training y data."""
        return self.trainy_mean_

    def get_ystd(self):
        """Returns the standard deviation of the training
        y data."""
        return self.trainy_std_

    def get_xdata(self):
        """Returns all xdata as a single array."""
        return self.xdata_

    def get_ydata(self):
        """Returns all ydata as a single array."""
        return self.ydata_
