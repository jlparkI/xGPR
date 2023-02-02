"""This module describes the OfflineDataset class.

The OfflineDataset class bundles together attributes for handling
a dataset stored on disk. These objects are generally created by
the build_dataset method of GPModelBaseclass and should not be
created directly by the user. The dataset can return a specific
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
    disk. It should be created using the build_dataset method of
    GPModelBaseclass, which performs a number of checks to ensure
    the data is valid before creating this object. Creating the object
    directly without using build_dataset bypasses those checks and is
    not recommended.

    Attributes:
        xfiles_ (list): A list of absolute filepaths to the locations
            of each xfile. Each xfile is a numpy array saved as a .npy.
        yfiles_ (list): A list of absolute filepaths to the locations
            of each yfile. Each yfile is a numpy array saved as a .npy.
        xdim_ (list): A list of length 2 (for 2d arrays) or 3 (for 3d
            arrays) where element 0 is the number of datapoints in the
            dataset, element 1 is shape[1] and element 2 is shape[2]
            (for 3d arrays for convolution kernels only).
        trainy_mean_ (float): The mean of the y-values.
        trainy_std_ (float): The standard deviation of the y-values.
        device (str): The current device.
        chunk_size (int): The largest allowed file size for this
            dataset. Should be checked and enforced by caller.
            Stored here for users of this object that need to know
            the largest file size in the dataset (in # datapoints).
        pretransformed_ (bool): If True, the data has already been
            "featurized" or transformed and the xfiles do not
            need to be run through the kernel. This can save time
            during fitting if an SSD is available. Default is False.
    """
    def __init__(self, xfiles,
                       yfiles,
                       xdim,
                       trainy_mean_,
                       trainy_std_,
                       device = "cpu",
                       chunk_size = 2000,
                       pretransformed = False):
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
            trainy_mean (float): The mean of the y-values.
            trainy_std (float): The standard deviation of the y-values.
            device (str): The current device.
            chunk_size (int): The largest allowed file size (in # datapoints)
                for this dataset. Should be checked and enforced by caller.
            pretransformed (bool): If True, the data has already been
                "featurized" or transformed and the xfiles do not
                need to be run through the kernel. This can save time
                during fitting if an SSD is available. Default is False.
        """
        super().__init__(pretransformed, xdim, device, chunk_size)
        self.xfiles_ = [os.path.abspath(f) for f in xfiles]
        self.yfiles_ = [os.path.abspath(f) for f in yfiles]
        self.trainy_mean_ = trainy_mean_
        self.trainy_std_ = trainy_std_

        self.mbatch_counter = 0
        self.mbatch_row = 0


    def get_chunked_data(self):
        """A generator that returns the data stored in each
        file in the data list in order with paired x and y
        data."""
        for xfile, yfile in zip(self.xfiles_, self.yfiles_):
            xchunk = self.array_loader(xfile)
            ychunk = self.array_loader(yfile).astype(self.dtype)
            if self.pretransformed:
                xchunk = xchunk.astype(self.dtype)
            ychunk -= self.trainy_mean_
            ychunk /= self.trainy_std_
            yield xchunk, ychunk


    def get_chunked_x_data(self):
        """A generator that returns the data stored in each
        file in the data list in order, retrieving x data
        only."""
        for xfile in self.xfiles_:
            xchunk = self.array_loader(xfile)
            if self.pretransformed:
                xchunk = xchunk.astype(self.dtype)
            yield xchunk

    def get_chunked_y_data(self):
        """A generator that returns the data stored in each
        file in the data list in order, retrieving y data
        only."""
        for yfile in self.yfiles_:
            ydata = self.array_loader(yfile).astype(self.dtype)
            yield (ydata - self.trainy_mean_) / self.trainy_std_



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
            xbatch = self.array_loader(self.xfiles_[self.mbatch_counter])
            ybatch = self.array_loader(self.yfiles_[self.mbatch_counter])
            cutoff = max(1, min(xbatch.shape[0], self.mbatch_row + batch_size - num_dpoints))
            num_dpoints += xbatch.shape[0]
            if len(xbatch.shape) == 3:
                xdata.append(xbatch[self.mbatch_row:cutoff,:,:])
            else:
                xdata.append(xbatch[self.mbatch_row:cutoff,:])
            ydata.append(ybatch[self.mbatch_row:cutoff])
            if cutoff < (xbatch.shape[0] - 1):
                self.mbatch_row = cutoff
                break

            self.mbatch_row = 0
            self.mbatch_counter += 1
            if self.mbatch_counter >= len(self.xfiles_):
                end_epoch = True
                self.mbatch_counter = 0

        xout = np.vstack(xdata)
        if self.pretransformed:
            xout = xout.astype(self.dtype)
        yout = np.concatenate(ydata).astype(self.dtype)
        if yout.shape[0] > batch_size:
            yout = yout[:batch_size]
            xout = xout[:batch_size,:]

        yout -= self.trainy_mean_
        yout /= self.trainy_std_
        if self.device == "gpu":
            xout = cp.asarray(xout)
            yout = cp.asarray(yout)
        return xout, yout, end_epoch


    def delete_dataset_files(self):
        """Deletes all the xfiles and yfiles saved by the dataset.
        Think twice before you do this! This is used by model
        classes only when data is pretransformed, in which case
        the temporarily saved pretransformed data can be deleted
        at the end of tuning / fitting."""
        for xfile, yfile in zip(self.xfiles_, self.yfiles_):
            os.remove(xfile)
            os.remove(yfile)
        self.xfiles_, self.yfiles_ = [], []


    def get_yfiles(self):
        """Returns a copy of the list of yfiles stored by the
        dataset."""
        return copy.copy(self.yfiles_)

    def get_xfiles(self):
        """Returns a copy of the list of xfiles stored by the
        dataset."""
        return copy.copy(self.xfiles_)

    def get_ymean(self):
        """Returns the mean of the training y data."""
        return self.trainy_mean_

    def get_ystd(self):
        """Returns the standard deviation of the training
        y data."""
        return self.trainy_std_
