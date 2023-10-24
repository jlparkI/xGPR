"""Tests the preconditioners, using a dataset we have used
in the past, to ensure that we can achieve a beta / lambda**2
less than an expected value."""
import unittest
import sys

import numpy as np

#TODO: Get rid of this path modification.
sys.path.append("..")
from utils.build_test_dataset import build_test_dataset
from utils.model_constructor import get_models


#A set of hyperparameters known to work well for our testing dataset
#that we can use as a default.
HPARAM = np.array([-0.67131348,  0.72078634, -1.00860899])


class CheckPreconditioners(unittest.TestCase):
    """Constructs an SRHT and Gaussian preconditioner and ensures both
    can achieve a beta / lambda_**2 as good as past results."""

    def test_srht_preconditioner(self):
        """Constructs an SRHT preconditioner and ensures it
        can achieve a beta / lambda_**2 similar to expected."""
        print("**********Testing SRHT**************")
        online_data, _ = build_test_dataset(conv_kernel = False)
        cpu_mod, gpu_mod = get_models("RBF", online_data, num_rffs=4100)
        _, ratio = cpu_mod.build_preconditioner(online_data,
            max_rank = 256, method = "srht", preset_hyperparams = HPARAM)
        self.assertTrue(ratio < 0.3)

        #If CUDA is available...
        if gpu_mod is not None:
            _, ratio = gpu_mod.build_preconditioner(online_data,
                max_rank = 256, method = "srht", preset_hyperparams = HPARAM)
            self.assertTrue(ratio < 0.3)
        print("\n\n\n")


    def test_gauss_preconditioner(self):
        """Constructs a non-SRHT preconditioner and ensures it
        can achieve a beta / lambda_**2 similar to expected."""
        print("*********Testing Non-SRHT**************")
        online_data, _ = build_test_dataset(conv_kernel = False)
        cpu_mod, gpu_mod = get_models("RBF", online_data, num_rffs=4100)
        _, ratio = cpu_mod.build_preconditioner(online_data,
            max_rank = 256, method = "gauss", preset_hyperparams = HPARAM)
        self.assertTrue(ratio < 0.3)

        #If CUDA is available...
        if gpu_mod is not None:
            _, ratio = gpu_mod.build_preconditioner(online_data,
                max_rank = 256, method = "gauss", preset_hyperparams = HPARAM)
            self.assertTrue(ratio < 0.3)
        print("\n\n\n")


if __name__ == "__main__":
    unittest.main()
