"""This module describes the OfflineDataset
for regression and classification.

The OfflineDataset bundles together attributes for handling
a dataset stored on disk. The dataset can return a specific
chunk of data from the list provided to it, can serve as a generator
and return all chunks in succession, or can provide a minibatch.
"""
import os
import copy

import numpy as np
try:
    import cupy as cp
except:
    pass

from .data_handling_baseclass import DatasetBaseclass



class OfflineDataset(DatasetBaseclass):
    """The OfflineDataset class handles datasets which are stored on
    disk. Only unique attributes not shared by parent class are
    described here.

    Attributes:
        _xfiles (list): A list of absolute filepaths to the locations
            of each xfile. Each xfile is a numpy array saved as a .npy.
        _yfiles (list): A list of absolute filepaths to the locations
            of each yfile. Each yfile is a numpy array saved as a .npy.
    """
    def __init__(self, xfiles,
                       yfiles,
                       xdim,
                       trainy_mean = 0.,
                       trainy_std = 1.,
                       max_class = 1,
                       device = "cpu",
                       chunk_size = 2000):
        """The class constructor for an OfflineDataset.

        Args:
            xfiles (list): A list of absolute filepaths to the locations
                of each xfile. Each xfile is a numpy array saved as a .npy.
            yfiles (list): A list of absolute filepaths to the locations
                of each yfile. Each yfile is a numpy array saved as a .npy.
            xdim (list): A list of length 2 (for 2d arrays) or 3 (for 3d
                arrays) where element 0 is the number of datapoints in the
                dataset, element 1 is shape[1] and element 2 is shape[2]
                (for 3d arrays for convolution kernels only).
            trainy_mean (float): The mean of the y-values. Only used for
                regression.
            trainy_std (float): The standard deviation of the y-values.
                Only used for regression.
            max_class (int): The largest category number in the data. Only
                used for classification.
            device (str): The current device.
            chunk_size (int): The largest allowed file size (in # datapoints)
                for this dataset. Should be checked and enforced by caller.
        """
        super().__init__(xdim, device, chunk_size, trainy_mean,
                trainy_std, max_class)
        self._xfiles = [os.path.abspath(f) for f in xfiles]
        self._yfiles = [os.path.abspath(f) for f in yfiles]


    def get_chunked_data(self):
        """A generator that returns the data stored in each
        file in the data list in order with paired x and y
        data."""
        for xfile, yfile in zip(self._xfiles, self._yfiles):
            xchunk = self.array_loader(xfile)
            ychunk = self.array_loader(yfile).astype(self.dtype)
            ychunk -= self._trainy_mean
            ychunk /= self._trainy_std
            yield xchunk, ychunk


    def get_chunked_x_data(self):
        """A generator that returns the data stored in each
        file in the data list in order, retrieving x data
        only."""
        for xfile in self._xfiles:
            xchunk = self.array_loader(xfile)
            yield xchunk

    def get_chunked_y_data(self):
        """A generator that returns the data stored in each
        file in the data list in order, retrieving y data
        only."""
        for yfile in self._yfiles:
            ydata = self.array_loader(yfile).astype(self.dtype)
            yield (ydata - self._trainy_mean) / self._trainy_std



    def get_next_minibatch(self, batch_size):
        """Returns the next minibatch in the training dataset.
        This is a little more complicated than get_chunked_data
        -- the minibatch size may not be the same size as a
        data chunk, so we can't simply loop over the data chunks.

        Args:
            batch_size (int): The number of datapoints expected for
                the batch.

        Returns:
            xout: A numpy or cupy array (depending on self.device)
                containing the xvalues.
            yout: A numpy or cupy array (depending on self.device)
                containing the yvalues.
            end_epoch (bool): If True, we are at the end of an epoch.
        """
        end_epoch = False
        xdata, ydata = [], []
        num_dpoints = 0
        while num_dpoints < batch_size:
            xbatch = self.array_loader(self._xfiles[self.mbatch_counter])
            ybatch = self.array_loader(self._yfiles[self.mbatch_counter])
            cutoff = max(1, min(xbatch.shape[0], self.mbatch_row + batch_size - num_dpoints))
            num_dpoints += xbatch.shape[0]

            xdata.append(xbatch[self.mbatch_row:cutoff,...])
            ydata.append(ybatch[self.mbatch_row:cutoff])
            if cutoff < (xbatch.shape[0] - 1):
                self.mbatch_row = cutoff
                break

            self.mbatch_row = 0
            self.mbatch_counter += 1
            if self.mbatch_counter >= len(self._xfiles):
                end_epoch = True
                self.mbatch_counter = 0

        xout = np.vstack(xdata)
        yout = np.concatenate(ydata).astype(self.dtype)
        if yout.shape[0] > batch_size:
            yout = yout[:batch_size]
            xout = xout[:batch_size,:]

        yout -= self._trainy_mean
        yout /= self._trainy_std
        if self.device == "gpu":
            xout = cp.asarray(xout)
            yout = cp.asarray(yout)
        return xout, yout, end_epoch


    def get_yfiles(self):
        """Returns a copy of the list of yfiles stored by the
        dataset."""
        return copy.copy(self._yfiles)

    def get_xfiles(self):
        """Returns a copy of the list of xfiles stored by the
        dataset."""
        return copy.copy(self._xfiles)
