"""Tests exact fitting (for small numbers of RFFs)."""
import sys
import unittest

import numpy as np

#TODO: Get rid of this path alteration
sys.path.append("..")
from utils.build_test_dataset import build_test_dataset
from utils.model_constructor import get_models
from utils.evaluate_model import evaluate_model

#A set of hyperparameters known to work well for our testing dataset
#that we can use as a default.
HPARAM = np.array([-0.67131348,  0.72078634, -1.00860899])

NUM_RFFS = 2100
RANDOM_SEED = 123


class CheckExactFit(unittest.TestCase):
    """Tests the exact fitting algorithm. We really just
    check to make sure there are no exceptions. For
    more extensive testing of performance using this
    and other fitting algorithms, see complete pipeline
    tests."""

    def test_exact_fit(self):
        """Test exact fitting."""
        online_data, _ = build_test_dataset(conv_kernel = False)
        cpu_mod, gpu_mod = get_models("RBF", online_data.get_xdim())
        cpu_mod.fitting_rffs = NUM_RFFS

        cpu_mod.fit(online_data,  random_seed = RANDOM_SEED, mode = "exact")

        if gpu_mod is not None:
            gpu_mod.fitting_rffs = NUM_RFFS
            gpu_mod.fit(online_data,  random_seed = RANDOM_SEED, mode = "exact")


if __name__ == "__main__":
    unittest.main()
