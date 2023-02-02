"""Tests (using preconditioned CG) to ensure that
pretransformation works correctly."""
import sys
import os
import unittest

import numpy as np

#TODO: Get rid of this path alteration
sys.path.append("..")
from utils.build_test_dataset import build_test_dataset
from utils.model_constructor import get_models
from utils.evaluate_model import evaluate_model

#A set of hyperparameters known to work well for our testing dataset
#that we can use as a default.
HPARAM = np.array([-0.67131348,  0.72078634, -1.00860899])

NUM_RFFS = 4100
RANDOM_SEED = 123


class CheckCGFit(unittest.TestCase):
    """Tests pretransformation (as used for CG)."""

    def test_pretransformed_cg(self):
        """Test using preconditioned cg with pretransformed data.
        Should easily fit in under 10 epochs."""
        online_data, _ = build_test_dataset(conv_kernel = False)
        cpu_mod, gpu_mod = get_models("RBF", online_data.get_xdim())
        cpu_mod.fitting_rffs = NUM_RFFS

        current_path = os.getcwd()
        pre_dataset = cpu_mod.pretransform_data(online_data,
                preset_hyperparams = HPARAM, random_seed = RANDOM_SEED,
                pretransform_dir = current_path)

        preconditioner, _ = cpu_mod.build_preconditioner(pre_dataset,
            max_rank = 256, method = "srht", preset_hyperparams = HPARAM)

        niter, _ = cpu_mod.fit(pre_dataset,  preconditioner = preconditioner,
                max_iter = 500, random_seed = RANDOM_SEED, run_diagnostics = True,
                tol = 1e-6,  mode = "cg")
        print(f"niter: {niter}")
        self.assertTrue(niter < 10)

        if gpu_mod is not None:
            gpu_mod.fitting_rffs = NUM_RFFS

            preconditioner, _ = gpu_mod.build_preconditioner(pre_dataset,
                max_rank = 256, method = "srht",
                preset_hyperparams = HPARAM)

            niter, _ = gpu_mod.fit(pre_dataset,  preconditioner = preconditioner,
                max_iter = 500, random_seed = RANDOM_SEED, run_diagnostics = True,
                tol = 1e-6,  mode = "cg")
            print(f"niter: {niter}")
            self.assertTrue(niter < 10)
        pre_dataset.delete_dataset_files()


if __name__ == "__main__":
    unittest.main()
