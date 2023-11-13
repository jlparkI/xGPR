"""Provides raw data loading services for other unit tests."""
import os

import numpy as np

from xGPR.data_handling.dataset_builder import build_regression_dataset

RANDOM_STATE = 123


def build_test_dataset(conv_kernel = False, xsuffix = "trainxvalues.npy",
        ysuffix = "trainyvalues.npy"):
    """Loads the test data provided with xGPR and converts it
    into an OfflineDataset object and an OnlineDataset object.
    Both are returned. Used by multiple other unit-tests.

    Args:
        conv_kernel (bool): If True, return the 3d array data
            for the convolution kernel. If False, return
            the 2d data for basic kernels.
        xsuffix (str): The expected ending for the file list that
            will be retrieved for x-data.
        ysuffix (str): The expected ending for the file list that
            will be retrieved for y-data.

    Returns:
        online_data (OnlineDataset): The raw data stored in memory.
        offline_data (OfflineDataset): The raw data stored on disk.
    """
    start_path = os.path.abspath(os.path.dirname(__file__))
    if not conv_kernel:
        os.chdir(os.path.join(start_path, "..", "..", "test_data"))
    else:
        os.chdir(os.path.join(start_path, "..", "..", "test_data", "conv_test"))
    xtrain_files = [f for f in os.listdir() if f.endswith(xsuffix)]
    ytrain_files = [f for f in os.listdir() if f.endswith(ysuffix)]
    xtrain_files.sort()
    ytrain_files.sort()

    offline_data = build_regression_dataset(xtrain_files,
                ytrain_files, chunk_size = 2000)
    xvalues, yvalues = [], []
    for xfile, yfile in zip(xtrain_files, ytrain_files):
        xvalues.append(np.load(xfile))
        yvalues.append(np.load(yfile))

    xvalues = np.vstack(xvalues)
    yvalues = np.concatenate(yvalues)
    online_data = build_regression_dataset(xvalues, yvalues, chunk_size = 2000)

    return online_data, offline_data



def build_traintest_split(conv_kernel = False, xsuffix = "trainxvalues.npy",
        ysuffix = "trainyvalues.npy"):
    """Loads the test data provided with xGPR and converts it
    into a train dataset and a test dataset.

    Args:
        conv_kernel (bool): If True, return the 3d array data
            for the convolution kernel. If False, return
            the 2d data for basic kernels.
        xsuffix (str): The expected ending for the file list that
            will be retrieved for x-data.
        ysuffix (str): The expected ending for the file list that
            will be retrieved for y-data.

    Returns:
        train_data (OnlineDataset): The training data.
        test_data (OnlineDataset): The test data.
    """
    start_path = os.path.abspath(os.path.dirname(__file__))
    if not conv_kernel:
        os.chdir(os.path.join(start_path, "..", "..", "test_data"))
    else:
        os.chdir(os.path.join(start_path, "..", "..", "test_data", "conv_test"))
    xtrain_files = [f for f in os.listdir() if f.endswith(xsuffix)]
    ytrain_files = [f for f in os.listdir() if f.endswith(ysuffix)]
    xtrain_files.sort()
    ytrain_files.sort()

    xvalues, yvalues = [], []
    for xfile, yfile in zip(xtrain_files, ytrain_files):
        xvalues.append(np.load(xfile))
        yvalues.append(np.load(yfile))

    xvalues = np.vstack(xvalues)
    yvalues = np.concatenate(yvalues)

    rng = np.random.default_rng(123)
    idx = rng.permutation(xvalues.shape[0])
    xvalues, yvalues = xvalues[idx,:], yvalues[idx]
    cutoff = int(0.75 * idx.shape[0])

    train_data = build_online_dataset(xvalues[:cutoff,...], yvalues[:cutoff], chunk_size = 2000)
    test_data = build_online_dataset(xvalues[cutoff:,...], yvalues[cutoff:], chunk_size = 2000)

    return train_data, test_data
