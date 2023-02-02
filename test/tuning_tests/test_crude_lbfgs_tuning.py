"""Tests the LBFGS tuning algorithm to ensure
we get performance on par with expectations."""
import sys
import unittest

#TODO: Get rid of this path alteration
sys.path.append("..")
from utils.build_test_dataset import build_test_dataset
from utils.model_constructor import get_models


class CheckCrudeLBFGSTuning(unittest.TestCase):
    """Tests the crude LBFGS tuning algorithm."""

    def test_lbfgs_tuning(self):
        """Test the LBFGS tuning algorithm using an
        RBF kernel for simplicity (tuning & fitting with other
        kernels is tested under the complete pipeline tests."""
        online_data, _ = build_test_dataset(conv_kernel = False)
        cpu_mod, gpu_mod = get_models("RBF", online_data.get_xdim())
        cpu_mod.training_rffs = 512
        _, _, best_score = cpu_mod.tune_hyperparams_crude_lbfgs(online_data,
                n_restarts = 3, max_iter = 50,
                random_seed = 123)
        self.assertTrue(best_score < 430)

        if gpu_mod is not None:
            gpu_mod.training_rffs = 512
            _, _, best_score = gpu_mod.tune_hyperparams_crude_lbfgs(online_data,
                n_restarts = 3, max_iter = 50,
                random_seed = 123)
            self.assertTrue(best_score < 430)


if __name__ == "__main__":
    unittest.main()
