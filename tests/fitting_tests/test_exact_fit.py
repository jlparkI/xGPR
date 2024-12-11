"""Tests exact fitting (for small numbers of RFFs)."""
import sys
import os
import unittest

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.build_test_dataset import build_test_dataset
from utils.build_classification_dataset import build_discriminant_traintest_split
from utils.model_constructor import get_models
from utils.evaluate_model import evaluate_model

#A set of hyperparameters known to work well for our testing dataset
#that we can use as a default.
HPARAM = np.array([np.log(np.sqrt(0.0767)),  np.log(0.358)])
DISCRIM_HPARAM = np.array([0., -0.75])

NUM_RFFS = 2100


class CheckExactFit(unittest.TestCase):
    """Tests the exact fitting algorithm. We really just
    check to make sure there are no exceptions. For
    more extensive testing of performance using this
    and other fitting algorithms, see complete pipeline
    tests."""

    def test_exact_fit(self):
        """Test exact fitting."""
        online_data, _ = build_test_dataset(conv_kernel = False)
        cpu_mod, gpu_mod = get_models("RBF", online_data, num_rffs = NUM_RFFS)

        cpu_mod.fit(online_data, mode = "exact")

        if gpu_mod is not None:
            gpu_mod.fit(online_data, mode = "exact")


if __name__ == "__main__":
    unittest.main()
