"""Tests the Direct_Fitting_Optimizer to ensure that
we can tune hyperparameters successfully using this
approach. GPU only."""
import unittest
import sys

import numpy as np

from xGPR.tuning_toolkit.direct_fitting_optimizer import DirectFittingOptimizer
#TODO: Remove this path modification
sys.path.append("..")
from utils.model_constructor import get_models
from utils.build_test_dataset import build_test_dataset
from utils.evaluate_model import evaluate_model



RANDOM_SEED = 123


class CheckDirectFittingOptimizer(unittest.TestCase):
    """Checks the DirectFittingOptimizer."""

    def test_tune_fitting_direct(self):
        """Checks the DirectFittingOptimizer."""
        _, offline_train = build_test_dataset(conv_kernel = False)
        online_test, offline_test = build_test_dataset(conv_kernel = False,
            xsuffix = "testxvalues.npy", ysuffix = "testyvalues.npy")
        _, gpu_mod = get_models("RBF", offline_train.get_xdim())
        if gpu_mod is None:
            raise ValueError("Only run this test if gpu is available.")

        gpu_mod.fitting_rffs = 4096
        #Using some generic bounds that make sense given the
        #"best" hyperparameters identified in tuning.

        bounds = np.array([ [-1.2, 0.0], [0.0,1.2], [-2,0.0] ])

        best_hparams = DirectFittingOptimizer(gpu_mod,
                optim_method = "Powell",
                train_dset = offline_train,
                bounds = bounds, max_feval = 50,
                validation_dset = offline_test,
                random_state = RANDOM_SEED, tol=1e-2,
                mode = "cg")

        preconditioner, _ = gpu_mod.build_preconditioner(offline_train,
                max_rank = 256, method = "srht",
                preset_hyperparams = best_hparams)
        gpu_mod.fit(offline_train,  preconditioner = preconditioner,
                max_iter = 500, random_seed = RANDOM_SEED,
                tol = 1e-6,  mode = "cg",
                preset_hyperparams = best_hparams)
        gpu_score = evaluate_model(gpu_mod, offline_train, online_test, False)
        self.assertTrue(gpu_score > 0.55)


if __name__ == "__main__":
    unittest.main()
