"""Tests the grid crude tuning algorithm to ensure
we get performance on par with expectations."""
import sys
import unittest

#TODO: Get rid of this path alteration
sys.path.append("..")
from utils.build_test_dataset import build_test_dataset
from utils.model_constructor import get_models


class CheckCrudegridTuning(unittest.TestCase):
    """Tests the grid crude tuning algorithm."""

    def test_grid_crude_tuning(self):
        """Test the grid crude tuning algorithm using an
        RBF kernel for simplicity (tuning & fitting with other
        kernels is tested under the complete pipeline tests."""
        online_data, _ = build_test_dataset(conv_kernel = False)
        cpu_mod, gpu_mod = get_models("RBF", online_data)
        _, _, best_score, _ = cpu_mod.tune_hyperparams_crude_grid(online_data,
                                n_gridpoints = 40)
        self.assertTrue(best_score < 430)

        if gpu_mod is not None:
            _, _, best_score, _ = gpu_mod.tune_hyperparams_crude_grid(online_data,
                                n_gridpoints = 40)
            self.assertTrue(best_score < 430)


if __name__ == "__main__":
    unittest.main()
