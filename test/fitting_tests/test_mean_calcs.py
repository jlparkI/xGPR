"""Tests the dataset mean calculation for discriminant classifiers."""
import sys
import unittest

import numpy as np

#TODO: Get rid of this path alteration
sys.path.append("..")
from utils.build_classification_dataset import build_discriminant_traintest_split
from utils.model_constructor import get_discriminant_models

#A set of hyperparameters known to work well for our testing dataset
#that we can use as a default.
DISCRIM_HPARAM = np.array([0., -0.75])

NUM_RFFS = 2100


class CheckDiscriminantMeanCalcs(unittest.TestCase):
    """Tests that the discriminant can calculate means
    and targets correctly."""

    def test_mean_calcs(self):
        """Test the mean calcs for the discriminant.
        Currently run on CPU only for simplicity."""
        online_data, _ = build_discriminant_traintest_split()
        cpu_mod, _ = get_discriminant_models("RBF", online_data,
                num_rffs = NUM_RFFS)

        cpu_mod.set_hyperparams(DISCRIM_HPARAM, online_data)
        x_mean, targets = cpu_mod._get_targets(online_data)
        self.assertTrue(cpu_mod.n_classes == 3)

        xvals, yvals = online_data.get_xdata(), online_data.get_ydata()

        true_features = cpu_mod.kernel.transform_x(xvals)
        true_mean = true_features.mean(axis=0)
        self.assertTrue(np.allclose(true_mean, x_mean))

        for i in range(cpu_mod.n_classes):
            target = true_features[yvals==i,...].mean(axis=0)
            self.assertTrue(np.allclose(target, targets[:,i]))


if __name__ == "__main__":
    unittest.main()
