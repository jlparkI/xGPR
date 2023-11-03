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
        models = get_models("RBF", online_data)
        for mod in models:
            if mod is None:
                continue
            mod.verbose = False
            bounds = np.array([[-3,0], [-3,0]])
            for tuning_method, nmll_method, max_iter in [("Powell", "exact", 50),
                    ("bayes", "approximate", 35)]:
                _, niter, best_score = mod.tune_hyperparams_direct(online_data,
                        tuning_method = tuning_method, n_restarts=3,
                        max_iter = max_iter, random_seed = 123,
                        nmll_method = nmll_method, nmll_rank = 256,
                        bounds = bounds)
                print(mod.get_hyperparams())
                print("*************")
                print(f"{tuning_method}, {mod.device}, {nmll_method}")
                print(f"Best score: {best_score}")
                print(f"{niter} iterations\n\n\n")
                self.assertTrue(best_score < 430)


if __name__ == "__main__":
    unittest.main()
