"""Tests the main tuning algorithm to ensure
we get performance on par with expectations."""
import sys
import os
import unittest

import numpy as np

from xGPR import xGPRegression as xGPReg
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.build_test_dataset import build_test_dataset
from utils.model_constructor import get_models


class CheckTuning(unittest.TestCase):
    """Tests the tuning algorithm."""

    def test_tuning(self):
        """Test the tuning algorithm using an
        RBF kernel for simplicity (tuning & fitting with other
        kernels is tested under the complete pipeline tests."""
        online_data, _ = build_test_dataset(conv_kernel = False)
        models = [m for m in get_models("RBF", online_data) if m is not None]

        for mod in models:
            mod.verbose = False
            for tuning_method, nmll_method, max_iter in [("Nelder-Mead", "exact", 100),
                        ("Powell", "exact", 100),
                        ("L-BFGS-B", "exact", 100)]:
                _, niter, best_score = mod.tune_hyperparams(online_data,
                        tuning_method = tuning_method, n_restarts=1,
                        starting_hyperparams = np.array([0., 0.]),
                        max_iter = max_iter, nmll_method = nmll_method)

                print(mod.get_hyperparams())
                print("*************")
                print(f"{tuning_method}, {mod.device}, {nmll_method}")
                print(f"Best score: {best_score}")
                print(f"{niter} iterations\n\n\n")
                self.assertTrue(best_score < 430)


if __name__ == "__main__":
    unittest.main()
