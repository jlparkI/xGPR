"""Tests cg fitting for both regression and discriminants."""
import sys
import os
import unittest

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.build_test_dataset import build_test_dataset
from utils.build_classification_dataset import build_discriminant_traintest_split
from utils.model_constructor import get_models
from utils.evaluate_model import evaluate_model

#Sets of hyperparameters known to work well for our testing dataset
#that we can use as a default. HPARAM is for regression only.
HPARAM = np.array([np.log(np.sqrt(0.0767)),  np.log(0.358)])
DISCRIM_HPARAM = np.array([0., -0.75])


NUM_RFFS = 8500


class CheckCGFit(unittest.TestCase):
    """Tests the conjugate gradients algorithm."""

    def test_preconditioned_cg(self):
        """Test using preconditioned cg, which should easily fit
        in under 10 epochs."""
        online_data, _ = build_test_dataset(conv_kernel = False)
        cpu_mod, gpu_mod = get_models("RBF", online_data, num_rffs = NUM_RFFS)

        cpu_mod.set_hyperparams(HPARAM, online_data)
        preconditioner, _ = cpu_mod.build_preconditioner(online_data,
            max_rank = 256, method = "srht")

        niter, _ = cpu_mod.fit(online_data,  preconditioner = preconditioner,
                max_iter = 500, run_diagnostics = True,
                tol = 1e-6,  mode = "cg")
        print(f"CPU, niter: {niter}")
        self.assertTrue(niter < 10)

        if gpu_mod is not None:
            gpu_mod.set_hyperparams(HPARAM, online_data)
            preconditioner, _ = gpu_mod.build_preconditioner(online_data,
                max_rank = 256, method = "srht")

            niter, _ = gpu_mod.fit(online_data,  preconditioner = preconditioner,
                max_iter = 500, run_diagnostics = True,
                tol = 1e-6,  mode = "cg")
            print(f"Cuda, niter: {niter}")
            self.assertTrue(niter < 10)


    def test_autoselect_cg(self):
        """Test using cg when the software automatically selects the
        max_rank."""
        online_data, _ = build_test_dataset(conv_kernel = False)
        cpu_mod, gpu_mod = get_models("RBF", online_data, num_rffs = NUM_RFFS)

        cpu_mod.set_hyperparams(HPARAM, online_data)
        niter, _ = cpu_mod.fit(online_data,
                max_iter = 500, run_diagnostics = True,
                tol = 1e-6,  mode = "cg")
        print(f"CPU autoselected preconditioning, niter: {niter}")
        self.assertTrue(niter < 10)

        if gpu_mod is not None:
            gpu_mod.set_hyperparams(HPARAM, online_data)
            niter, _ = gpu_mod.fit(online_data,
                max_iter = 500, run_diagnostics = True,
                tol = 1e-6,  mode = "cg")
            print(f"Cuda autoselected preconditioning, niter: {niter}")
            self.assertTrue(niter < 10)


if __name__ == "__main__":
    unittest.main()
