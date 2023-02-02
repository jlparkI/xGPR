"""Tests LBFGS fitting."""
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
HPARAM = np.array([-0.6826397, 0.695573, -0.98540819])

NUM_RFFS = 4100
RANDOM_SEED = 123


class CheckLBFGSFit(unittest.TestCase):
    """Tests LBFGS fitting."""

    def test_lbfgs(self):
        """Test using LBFGS, which should easily fit in under
        100 epochs."""
        online_data, _ = build_test_dataset(conv_kernel = False)
        cpu_mod, gpu_mod = get_models("RBF", online_data.get_xdim())
        cpu_mod.fitting_rffs = NUM_RFFS

        niter, _ = cpu_mod.fit(online_data,
                max_iter = 500, random_seed = RANDOM_SEED, run_diagnostics = True,
                mode = "lbfgs", preset_hyperparams = HPARAM)
        print(f"niter: {niter}")
        self.assertTrue(niter < 100)

        if gpu_mod is not None:
            gpu_mod.fitting_rffs = NUM_RFFS

            niter, _ = gpu_mod.fit(online_data,
                max_iter = 500, random_seed = RANDOM_SEED, run_diagnostics = True,
                tol = 1e-6,  mode = "lbfgs", preset_hyperparams = HPARAM)
            print(f"niter: {niter}")
            self.assertTrue(niter < 100)


if __name__ == "__main__":
    unittest.main()
