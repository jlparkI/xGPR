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
        _sequence_lengths: Either None or a list of absolute filepaths
            to the locations of each .npy file containing the length
            of the sequence / graph for the corresponding datapoint.
    """
    def __init__(self, xfiles,
                       yfiles,
                       sequence_lengths,
                       xdim,
                       trainy_mean = 0.,
                       trainy_std = 1.,
                       max_class = 1,
                       chunk_size = 2000):
        """The class constructor for an OfflineDataset.

        Args:
            xfiles (list): A list of absolute filepaths to the locations
                of each xfile. Each xfile is a numpy array saved as a .npy.
            yfiles (list): A list of absolute filepaths to the locations
                of each yfile. Each yfile is a numpy array saved as a .npy.
            sequence_lengths: Either None or a list of absolute filepaths
                to the locations of each .npy file containing the length
                of the sequence / graph for the corresponding datapoint.
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
        super().__init__(xdim, chunk_size, trainy_mean,
                trainy_std, max_class)
        self._xfiles = [os.path.abspath(f) for f in xfiles]
        self._yfiles = [os.path.abspath(f) for f in yfiles]
        if sequence_lengths is not None:
            self._sequence_lengths = [os.path.abspath(f) for f in sequence_lengths]
        else:
            self._sequence_lengths = None


    def get_chunked_data(self):
        """A generator that returns the data stored in each
        file in the data list in order with paired x and y
        data."""
        if self._sequence_lengths is None:
            for xfile, yfile in zip(self._xfiles, self._yfiles):
                xchunk = np.load(xfile)
                ychunk = np.load(yfile).astype(np.float64)
                ychunk -= self._trainy_mean
                ychunk /= self._trainy_std
                yield xchunk, ychunk, None
        else:
            for xfile, yfile, lfile in zip(self._xfiles, self._yfiles,
                    self._sequence_lengths):
                xchunk, lchunk = np.load(xfile), np.load(lfile)
                ychunk = np.load(yfile).astype(np.float64)
                ychunk -= self._trainy_mean
                ychunk /= self._trainy_std
                yield xchunk, ychunk, lchunk



    def get_chunked_x_data(self):
        """A generator that returns the data stored in each
        file in the data list in order, retrieving x data
        and sequence length only."""
        if self._sequence_lengths is None:
            for xfile in self._xfiles:
                yield np.load(xfile), None
        else:
            for xfile, lfile in zip(self._xfiles, self._sequence_lengths):
                yield np.load(xfile), np.load(lfile)
