"""Tests the direct tuning algorithm to ensure
we get performance on par with expectations."""
import sys
import unittest

import numpy as np

from xGPR.xGP_Regression import xGPRegression as xGPReg
#TODO: Get rid of this path alteration
sys.path.append("..")
from utils.build_test_dataset import build_test_dataset
from utils.model_constructor import get_models


class CheckDirectTuning(unittest.TestCase):
    """Tests the NM tuning algorithm."""

    def test_direct_tuning(self):
        """Test the direct tuning algorithm using an
        RBF kernel for simplicity (tuning & fitting with other
        kernels is tested under the complete pipeline tests."""
        online_data, _ = build_test_dataset(conv_kernel = False)
        cpu_mod, gpu_mod = get_models("RBF", online_data)
        bounds = np.array([[-2,0],    [-3,0]])
        _, niter, best_score = cpu_mod.tune_hyperparams_fine_direct(online_data,
                optim_method = "Powell",
                max_iter = 50, bounds = bounds,
                random_seed = 123, nmll_rank = 128,
                nmll_iter = 500, nmll_tol = 1e-5,
                preconditioner_mode = "srht")
        print(f"Best score: {best_score}")
        print(f"{niter} iterations")
        self.assertTrue(best_score < 430)

        if gpu_mod is not None:
            _, niter, best_score = gpu_mod.tune_hyperparams_fine_direct(online_data,
                optim_method = "Powell",
                max_iter = 50, bounds = bounds,
                random_seed = 123,
                nmll_rank = 128, nmll_iter = 500,
                nmll_tol = 1e-5,
                preconditioner_mode = "srht")
            print(f"Best score: {best_score}")
            print(f"{niter} iterations")
            self.assertTrue(best_score < 430)


if __name__ == "__main__":
    unittest.main()
