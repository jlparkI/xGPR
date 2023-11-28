"""Tests LBFGS fitting."""
import sys
import unittest

import numpy as np

#TODO: Get rid of this path alteration
sys.path.append("..")
from utils.build_test_dataset import build_test_dataset
from utils.build_classification_dataset import build_discriminant_traintest_split
from utils.model_constructor import get_models, get_discriminant_models
from utils.evaluate_model import evaluate_model

#A set of hyperparameters known to work well for our testing dataset
#that we can use as a default. HPARAM is regression only.
HPARAM = np.array([np.log(np.sqrt(0.0767)),  np.log(0.358)])
DISCRIM_HPARAM = np.array([0., -0.75])

NUM_RFFS = 4100


class CheckLBFGSFit(unittest.TestCase):
    """Tests LBFGS fitting."""

    def test_lbfgs(self):
        """Test using LBFGS, which should easily fit in under
        150 epochs."""
        online_data, _ = build_test_dataset(conv_kernel = False)
        cpu_mod, gpu_mod = get_models("RBF", online_data, num_rffs = NUM_RFFS)

        cpu_mod.set_hyperparams(HPARAM, online_data)
        niter, _ = cpu_mod.fit(online_data,
                max_iter = 500, run_diagnostics = True,
                mode = "lbfgs")
        print(f"niter: {niter}")
        self.assertTrue(niter < 150)

        if gpu_mod is not None:
            gpu_mod.set_hyperparams(HPARAM, online_data)
            niter, _ = gpu_mod.fit(online_data,
                max_iter = 500, run_diagnostics = True,
                tol = 1e-6,  mode = "lbfgs")
            print(f"niter: {niter}")
            self.assertTrue(niter < 150)


    def test_lbfgs_discriminant(self):
        """Test using LBFGS, which should easily fit in under
        150 epochs for the discriminant classifier."""
        online_data, _ = build_discriminant_traintest_split()
        cpu_mod, gpu_mod = get_discriminant_models("RBF", online_data,
                num_rffs = NUM_RFFS)

        cpu_mod.set_hyperparams(DISCRIM_HPARAM, online_data)
        niter = cpu_mod.fit(online_data,
                max_iter = 500, run_diagnostics = True,
                mode = "lbfgs")
        print(f"niter: {niter}")
        self.assertTrue(niter < 150)

        if gpu_mod is not None:
            gpu_mod.set_hyperparams(DISCRIM_HPARAM, online_data)
            niter = gpu_mod.fit(online_data,
                max_iter = 500, run_diagnostics = True,
                tol = 1e-6,  mode = "lbfgs")
            print(f"niter: {niter}")
            self.assertTrue(niter < 150)


if __name__ == "__main__":
    unittest.main()
