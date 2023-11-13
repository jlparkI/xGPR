"""Tests cg fitting to ensure success both with and without
preconditioning."""
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
HPARAM = np.array([np.log(np.sqrt(0.0767)),  np.log(0.358)])

NUM_RFFS = 4100


class CheckCGFit(unittest.TestCase):
    """Tests the conjugate gradients algorithm."""

    def test_preconditioned_cg(self):
        """Test using preconditioned cg, which should easily fit
        in under 10 epochs."""
        online_data, _ = build_test_dataset(conv_kernel = False)
        cpu_mod, gpu_mod = get_models("RBF", online_data, num_rffs = NUM_RFFS)

        preconditioner, _ = cpu_mod.build_preconditioner(online_data,
            max_rank = 256, method = "srht", preset_hyperparams = HPARAM)

        niter, _ = cpu_mod.fit(online_data,  preconditioner = preconditioner,
                max_iter = 500, run_diagnostics = True,
                tol = 1e-6,  mode = "cg")
        print(f"niter: {niter}")
        self.assertTrue(niter < 10)

        if gpu_mod is not None:
            preconditioner, _ = gpu_mod.build_preconditioner(online_data,
                max_rank = 256, method = "srht",
                preset_hyperparams = HPARAM)

            niter, _ = gpu_mod.fit(online_data,  preconditioner = preconditioner,
                max_iter = 500, run_diagnostics = True,
                tol = 1e-6,  mode = "cg")
            print(f"niter: {niter}")
            self.assertTrue(niter < 10)


    def test_nonpreconditioned_cg(self):
        """Test using non-preconditioned cg."""
        online_data, _ = build_test_dataset(conv_kernel = False)
        cpu_mod, gpu_mod = get_models("RBF", online_data, num_rffs = NUM_RFFS)
        cpu_mod.verbose = False

        niter, _ = cpu_mod.fit(online_data,
                max_iter = 500, run_diagnostics = True,
                tol = 1e-6,  mode = "cg", preset_hyperparams = HPARAM)
        print(f"No preconditioning, niter: {niter}")
        self.assertTrue(niter < 80)

        if gpu_mod is not None:
            gpu_mod.verbose = False
            niter, _ = gpu_mod.fit(online_data,
                max_iter = 500, run_diagnostics = True,
                tol = 1e-6,  mode = "cg", preset_hyperparams = HPARAM)
            print(f"No preconditioning, niter: {niter}")
            self.assertTrue(niter < 80)


if __name__ == "__main__":
    unittest.main()
