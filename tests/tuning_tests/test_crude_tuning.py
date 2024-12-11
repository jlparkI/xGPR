"""Tests the crude tuning algorithm to ensure
we get performance on par with expectations."""
import sys
import os
import unittest

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.build_test_dataset import build_test_dataset
from utils.model_constructor import get_models


class CheckCrudeTuning(unittest.TestCase):
    """Tests the crude tuning algorithm."""

    def test_crude_tuning(self):
        """Test the crude tuning algorithm using an
        RBF kernel for simplicity (tuning & fitting with other
        kernels is tested under the complete pipeline tests."""
        online_data, _ = build_test_dataset(conv_kernel = False)
        cpu_mod, gpu_mod = get_models("RBF", online_data)
        cpu_mod.num_rffs = 512
        _, _, best_score = cpu_mod.tune_hyperparams_crude(online_data)
        self.assertTrue(best_score < 430)

        if gpu_mod is not None:
            gpu_mod.num_rffs = 512
            _, _, best_score = gpu_mod.tune_hyperparams_crude(online_data)
            self.assertTrue(best_score < 430)


if __name__ == "__main__":
    unittest.main()
