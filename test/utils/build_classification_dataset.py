"""Provides data for classification-specific unit tests. We test
all the kernels in regression-specific unit tests, so for classification,
we use a fixed-vector dataset only with the RBF kernel. The wine
dataset is used here as a (horrifically) simple test."""
import sklearn
from sklearn import datasets
import numpy as np

from xGPR.data_handling.dataset_builder import build_classification_dataset

RANDOM_STATE = 123


def build_classifification_traintest_split():
    """Loads the wine dataset from sklearn and returns a train-test
    split of the data.

    Returns:
        online_data (OnlineDataset): The raw data stored in memory.
    """
    xvalues, yvalues = sklearn.datasets.load_wine(return_X_y = True)
    rng = np.random.default_rng(123)
    idx = rng.permutation(xvalues.shape[0])
    xvalues, yvalues = xvalues[idx,:], yvalues[idx]
    cutoff = int(0.75 * idx.shape[0])

    train_data = build_classification_dataset(xvalues[:cutoff,...],
            yvalues[:cutoff], chunk_size = 2000)
    test_data = build_classification_dataset(xvalues[cutoff:,...],
            yvalues[cutoff:], chunk_size = 2000)

    return train_data, test_data
