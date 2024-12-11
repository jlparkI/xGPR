"""Tests the preconditioners, using a dataset we have used
in the past, to ensure that we can achieve a beta / lambda**2
less than an expected value."""
import unittest
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.build_test_dataset import build_test_dataset
from utils.model_constructor import get_models


#A set of hyperparameters known to work well for our testing dataset
#that we can use as a default.
HPARAM = np.array([-1.39209982,  -1.00860899])


class CheckPreconditioners(unittest.TestCase):
    """Constructs an SRHT and Gaussian preconditioner and ensures both
    can achieve a beta / lambda_**2 as good as past results."""

    def test_srht_preconditioner(self):
        """Constructs an SRHT preconditioner and ensures it
        can achieve a beta / lambda_**2 similar to expected."""
        print("**********Testing SRHT**************")
        online_data, _ = build_test_dataset(conv_kernel = False)
        cpu_mod, gpu_mod = get_models("RBF", online_data, num_rffs=4100)
        cpu_mod.set_hyperparams(HPARAM, online_data)
        _, cpu_ratio = cpu_mod.build_preconditioner(online_data,
            max_rank = 256, method = "srht")
        print(f"CPU ratio: {cpu_ratio}")
        self.assertTrue(cpu_ratio < 0.3)

        #If CUDA is available...
        if gpu_mod is not None:
            gpu_mod.set_hyperparams(HPARAM, online_data)
            _, gpu_ratio = gpu_mod.build_preconditioner(online_data,
                max_rank = 256, method = "srht")
            print(f"GPU ratio: {gpu_ratio}")
            self.assertTrue(gpu_ratio < 0.3)
    

    def test_srht2_preconditioner(self):
        """Constructs an SRHT2 preconditioner and ensures it
        can achieve a beta / lambda_**2 similar to expected."""
        print("**********Testing SRHT2**************")

        online_data, _ = build_test_dataset(conv_kernel = False)
        cpu_mod, gpu_mod = get_models("RBF", online_data, num_rffs=4100)
        cpu_mod.set_hyperparams(HPARAM, online_data)
        _, cpu_ratio = cpu_mod.build_preconditioner(online_data,
            max_rank = 256, method = "srht_2")
        print(f"CPU ratio: {cpu_ratio}")
        self.assertTrue(cpu_ratio < 0.4)

        #If CUDA is available...
        if gpu_mod is not None:
            gpu_mod.set_hyperparams(HPARAM, online_data)
            _, gpu_ratio = gpu_mod.build_preconditioner(online_data,
                max_rank = 256, method = "srht_2")
            print(f"GPU ratio: {gpu_ratio}")
            self.assertTrue(gpu_ratio < 0.4)



    def test_sampled_preconditioner(self):
        """Constructs a sampled preconditioner and determines whether
        the estimated ratio compares favorably to exact."""
        print("**********Testing sampled preconditioner**************")
        online_data, _ = build_test_dataset(conv_kernel = False)
        cpu_mod, gpu_mod = get_models("RBF", online_data, num_rffs=4100)
        cpu_mod.set_hyperparams(HPARAM, online_data)

        _, cpu_ratio = cpu_mod.build_preconditioner(online_data,
            max_rank = 64, method = "srht")
        sampled_ratio = cpu_mod._check_rank_ratio(online_data, 0.5,
                max_rank = 64)
        self.assertTrue((sampled_ratio / cpu_ratio) < 1.5)
        print(f"CPU exact {cpu_ratio}, sampled {sampled_ratio}")

        #If CUDA is available...
        if gpu_mod is not None:
            gpu_mod.set_hyperparams(HPARAM, online_data)
            _, gpu_ratio = gpu_mod.build_preconditioner(online_data,
                max_rank = 64, method = "srht")
            sampled_ratio = gpu_mod._check_rank_ratio(online_data, 0.5,
                    max_rank = 64)
            print(f"GPU exact {gpu_ratio}, sampled {sampled_ratio}")
            self.assertTrue((sampled_ratio / gpu_ratio) < 1.5)


if __name__ == "__main__":
    unittest.main()
