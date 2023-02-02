"""Tests the Adam tuning algorithm to ensure
we get performance on par with expectations."""
import sys
import unittest

#TODO: Get rid of this path alteration
sys.path.append("..")
from utils.build_test_dataset import build_test_dataset
from utils.model_constructor import get_models


class CheckSGDTuning(unittest.TestCase):
    """Tests the SGD tuning algorithm."""

    def test_sgd_tuning(self):
        """Test the sgd tuning algorithm using an
        RBF kernel for simplicity (tuning & fitting with other
        kernels is tested under the complete pipeline tests.
        sgd performs worse than other tuning algorithms so we
        'grade it on a curve' -- it is a biased estimator
        of the full gradient."""
        online_data, _ = build_test_dataset(conv_kernel = False)
        cpu_mod, gpu_mod = get_models("RBF", online_data.get_xdim())
        cpu_mod.training_rffs = 512
        _, _, best_score = cpu_mod.tune_hyperparams_crude_sgd(online_data,
                minibatch_size = 100,
                random_seed = 123, n_restarts = 5, learn_rate = 0.02,
                n_epochs = 20, nmll_method = "exact")
        print(f"Best score: {best_score}")
        self.assertTrue(best_score < 430)

        if gpu_mod is not None:
            gpu_mod.training_rffs = 512
            _, _, best_score = gpu_mod.tune_hyperparams_crude_sgd(online_data,
                minibatch_size = 100,
                random_seed = 123, n_restarts = 5, learn_rate = 0.02,
                n_epochs = 20, nmll_method = "exact")
            self.assertTrue(best_score < 430)
            print(f"Best score: {best_score}")


if __name__ == "__main__":
    unittest.main()
