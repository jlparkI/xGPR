"""Provides data for classification-specific unit tests. We test
all the kernels in regression-specific unit tests, so for classification,
we use a fixed-vector dataset only with the RBF kernel. The wine
dataset is used here as a (horrifically) simple test."""
import sklearn
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np

from xGPR.data_handling.dataset_builder import build_classification_dataset

RANDOM_STATE = 123


def build_discriminant_traintest_split():
    """Loads the wine dataset from sklearn and returns a train-test
    split of the data.

    Returns:
        online_data (OnlineDataset): The raw data stored in memory.
    """
    xvalues, yvalues = sklearn.datasets.load_wine(return_X_y = True)
    scaler = StandardScaler()

    #Scaling train and test data together is ordinarily not great, but
    #this is just for testing purposes.
    xvalues = scaler.fit_transform(xvalues)

    rng = np.random.default_rng(123)
    idx = rng.permutation(xvalues.shape[0])
    xvalues, yvalues = xvalues[idx,:], yvalues[idx]
    cutoff = int(0.75 * idx.shape[0])

    if len(xvalues.shape) == 2:
        sequence_lengths = None
        train_data = build_classification_dataset(xvalues[:cutoff,...],
                yvalues[:cutoff], chunk_size = 2000)
        test_data = build_classification_dataset(xvalues[cutoff:,...],
                yvalues[cutoff:], chunk_size = 2000)

    else:
        sequence_lengths = np.full(xvalues.shape[0], xvalues.shape[1]).astype(np.int32)
        train_data = build_classification_dataset(xvalues[:cutoff,...],
                yvalues[:cutoff], sequence_lengths[:cutoff], chunk_size = 2000)
        test_data = build_classification_dataset(xvalues[cutoff:,...],
                yvalues[cutoff:], sequence_lengths[cutoff:], chunk_size = 2000)


    return train_data, test_data
