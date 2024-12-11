"""Tests dataset construction and ensures that y_mean, y_std
are calculated correctly."""
import sys
import os
import unittest

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.build_test_dataset import build_test_dataset



class CheckDatasetConstruction(unittest.TestCase):
    """Tests construction of dataset objects."""

    def test_dataset_builders(self):
        """Test the dataset builders."""
        test_online_dataset, test_offline_dataset = build_test_dataset(conv_kernel = False,
            xsuffix = "testxvalues.npy", ysuffix = "testyvalues.npy")
        train_online_dataset, train_offline_dataset = build_test_dataset(conv_kernel = False)

        #Access protected class members directly
        test_ymean = np.mean(test_online_dataset._ydata)
        test_ystd = np.std(test_online_dataset._ydata)
        train_ymean = np.mean(train_online_dataset._ydata)
        train_ystd = np.std(train_online_dataset._ydata)

        self.assertTrue(np.allclose(test_ymean, test_offline_dataset.get_ymean()))
        self.assertTrue(np.allclose(test_ystd, test_offline_dataset.get_ystd()))
        self.assertTrue(np.allclose(train_ymean, train_offline_dataset.get_ymean()))
        self.assertTrue(np.allclose(train_ystd, train_offline_dataset.get_ystd()))


        test_xdim = test_online_dataset._xdata.shape
        self.assertTrue(test_xdim == test_offline_dataset.get_xdim())
        self.assertTrue(test_xdim == test_online_dataset.get_xdim())

if __name__ == "__main__":
    unittest.main()
