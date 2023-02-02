"""Tests svrg fitting both with and without preconditioning."""
import sys
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


class CheckSVRGFit(unittest.TestCase):
    """Tests the SVRG algorithm."""

    def test_svrg_preconditioned(self):
        """Test using SVRG, which is slower than CG but
        should still fit this in a reasonably short period
        of time. The key thing that can go wrong with SVRG is
        the step size selection -- if this is not correct it
        can diverge. The user can select manually so here
        we are testing the autofitting procedure, which selects
        a step size for them. Crucial to ensure this works."""
        online_data, _ = build_test_dataset(conv_kernel = False)
        cpu_mod, gpu_mod = get_models("RBF", online_data.get_xdim())
        cpu_mod.fitting_rffs = NUM_RFFS

        preconditioner, _ = cpu_mod.build_preconditioner(online_data,
            max_rank = 256, method = "srht", preset_hyperparams = HPARAM)

        niter, _ = cpu_mod.fit(online_data,  preconditioner = preconditioner,
                max_iter = 500, random_seed = RANDOM_SEED, run_diagnostics = True,
                tol = 1e-6,  mode = "sgd")
        print(f"niter: {niter}")
        self.assertTrue(niter < 20)

        if gpu_mod is not None:
            gpu_mod.fitting_rffs = NUM_RFFS

            preconditioner, _ = gpu_mod.build_preconditioner(online_data,
                max_rank = 256, method = "srht",
                preset_hyperparams = HPARAM)

            niter, _ = gpu_mod.fit(online_data,  preconditioner = preconditioner,
                max_iter = 500, random_seed = RANDOM_SEED, run_diagnostics = True,
                tol = 1e-6,  mode = "sgd")
            print(f"niter: {niter}")
            self.assertTrue(niter < 20)


    def test_svrg_no_preconditioner(self):
        """Test using SVRG without preconditioning, which is in
        general an EXTREMELY bad algorithm for fitting and can
        take hundreds of iterations to reach an acceptable
        result. Here we grade it 'on a curve' and just make
        sure it can reach 2e-1 in under 150 iterations (!!!).
        It is unlikely any user would ever use SVRG without
        preconditioning, but still useful to ensure it is
        functional."""
        online_data, _ = build_test_dataset(conv_kernel = False)
        cpu_mod, gpu_mod = get_models("RBF", online_data.get_xdim())
        cpu_mod.fitting_rffs = NUM_RFFS

        niter, _ = cpu_mod.fit(online_data,
                max_iter = 500, random_seed = RANDOM_SEED, run_diagnostics = True,
                tol = 2e-1,  mode = "sgd", preset_hyperparams = HPARAM)
        print(f"niter: {niter}")
        self.assertTrue(niter < 150)

        if gpu_mod is not None:
            gpu_mod.fitting_rffs = NUM_RFFS

            niter, _ = gpu_mod.fit(online_data,
                max_iter = 500, random_seed = RANDOM_SEED, run_diagnostics = True,
                tol = 2e-1,  mode = "sgd", preset_hyperparams = HPARAM)
            print(f"niter: {niter}")
            self.assertTrue(niter < 150)


if __name__ == "__main__":
    unittest.main()
