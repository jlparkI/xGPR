"""Tests lsr1 fitting to ensure success both with and without
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
RANDOM_SEED = 123


class CheckCGFit(unittest.TestCase):
    """Tests the conjugate gradients algorithm."""

    def test_preconditioned_lsr1(self):
        """Test using preconditioned lsr1, which should easily fit
        in under 10 epochs."""
        online_data, _ = build_test_dataset(conv_kernel = False)
        cpu_mod, gpu_mod = get_models("RBF", online_data, num_rffs = NUM_RFFS)

        preconditioner, _ = cpu_mod.build_preconditioner(online_data,
            max_rank = 256, method = "srht", preset_hyperparams = HPARAM)

        niter, _ = cpu_mod.fit(online_data,  preconditioner = preconditioner,
                max_iter = 500, random_seed = RANDOM_SEED, run_diagnostics = True,
                tol = 1e-6,  mode = "lsr1")
        print(f"LSR1, niter: {niter}")
        self.assertTrue(niter < 10)

        if gpu_mod is not None:
            preconditioner, _ = gpu_mod.build_preconditioner(online_data,
                max_rank = 256, method = "srht",
                preset_hyperparams = HPARAM)

            niter, _ = gpu_mod.fit(online_data,  preconditioner = preconditioner,
                max_iter = 500, random_seed = RANDOM_SEED, run_diagnostics = True,
                tol = 1e-6,  mode = "lsr1")
            print(f"LSR1, niter: {niter}")
            self.assertTrue(niter < 10)


if __name__ == "__main__":
    unittest.main()
