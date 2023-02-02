"""Tests the fine bayes tuning algorithm to ensure
we get performance on par with expectations."""
import sys
import unittest

import numpy as np

from xGPR.xGP_Regression import xGPRegression as xGPReg
#TODO: Get rid of this path alteration
sys.path.append("..")
from utils.build_test_dataset import build_test_dataset
from utils.model_constructor import get_models


class CheckFBTuning(unittest.TestCase):
    """Tests the fine bayes tuning algorithm."""

    def test_fb_tuning(self):
        """Test the fine bayes tuning algorithm using an
        RBF kernel for simplicity (tuning & fitting with other
        kernels is tested under the complete pipeline tests."""
        online_data, _ = build_test_dataset(conv_kernel = False)
        cpu_mod, gpu_mod = get_models("RBF", online_data.get_xdim())
        cpu_mod.training_rffs = 512
        bounds = np.array([[-2,0],    [0,2],    [-3,0]])
        cpu_mod.kernel = None
        _, niter, best_score = cpu_mod.tune_hyperparams_fine_bayes(online_data,
                max_bayes_iter = 70, bounds = bounds,
                random_seed = 123, nmll_rank = 128,
                nmll_iter = 500, nmll_tol = 1e-6,
                preconditioner_mode = "srht")
        print(f"Best score: {best_score}")
        print(f"{niter} iterations")
        self.assertTrue(best_score < 430)

        if gpu_mod is not None:
            gpu_mod.training_rffs = 512
            gpu_mod.kernel = None
            _, niter, scores = gpu_mod.tune_hyperparams_fine_bayes(online_data,
                max_bayes_iter = 70, bounds = bounds,
                random_seed = 123,
                nmll_rank = 128, nmll_iter = 500,
                nmll_tol = 1e-6,
                preconditioner_mode = "srht")
            print(f"Best score: {best_score}")
            print(f"{niter} iterations")
            self.assertTrue(best_score < 430)


if __name__ == "__main__":
    unittest.main()
