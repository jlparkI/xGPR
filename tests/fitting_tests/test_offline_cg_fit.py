"""Tests fitting with an offline dataset (to ensure results
are the same)."""
import sys
import os
import unittest

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.build_test_dataset import build_test_dataset
from utils.build_classification_dataset import build_discriminant_traintest_split
from utils.model_constructor import get_models, get_discriminant_models
from utils.evaluate_model import evaluate_model

#A set of hyperparameters known to work well for our testing dataset
#that we can use as a default.
HPARAM = np.array([np.log(np.sqrt(0.0767)),  np.log(0.358)])
DISCRIM_HPARAM = np.array([0., -0.75])

NUM_RFFS = 4100


class CheckCGFit(unittest.TestCase):
    """Tests the conjugate gradients algorithm."""


    def test_preconditioned_cg(self):
        """Test using preconditioned cg, which should easily fit
        in under 10 epochs."""
        _, offline_data = build_test_dataset(conv_kernel = False)
        cpu_mod, gpu_mod = get_models("RBF", offline_data, num_rffs = NUM_RFFS)

        cpu_mod.set_hyperparams(HPARAM, offline_data)
        preconditioner, _ = cpu_mod.build_preconditioner(offline_data,
            max_rank = 256, method = "srht")

        niter, _ = cpu_mod.fit(offline_data,  preconditioner = preconditioner,
                max_iter = 500, run_diagnostics = True,
                tol = 1e-6,  mode = "cg")
        print(f"niter: {niter}")
        self.assertTrue(niter < 10)

        if gpu_mod is not None:
            gpu_mod.set_hyperparams(HPARAM, offline_data)
            preconditioner, _ = gpu_mod.build_preconditioner(offline_data,
                max_rank = 256, method = "srht")

            niter, _ = gpu_mod.fit(offline_data,  preconditioner = preconditioner,
                max_iter = 500, run_diagnostics = True,
                tol = 1e-6,  mode = "cg")
            print(f"niter: {niter}")
            self.assertTrue(niter < 10)


    def test_autoselect_cg(self):
        """Test using preconditioner autoselect."""
        _, offline_data = build_test_dataset(conv_kernel = False)
        cpu_mod, gpu_mod = get_models("RBF", offline_data)
        cpu_mod.fitting_rffs = NUM_RFFS
        cpu_mod.verbose = False

        cpu_mod.set_hyperparams(HPARAM, offline_data)
        niter, _ = cpu_mod.fit(offline_data,
                max_iter = 500, run_diagnostics = True,
                tol = 1e-6,  mode = "cg")
        print(f"Autoselect preconditioning, niter: {niter}")
        self.assertTrue(niter < 80)

        if gpu_mod is not None:
            gpu_mod.fitting_rffs = NUM_RFFS
            gpu_mod.verbose = False
            gpu_mod.set_hyperparams(HPARAM, offline_data)
            niter, _ = gpu_mod.fit(offline_data,
                max_iter = 500, run_diagnostics = True,
                tol = 1e-6,  mode = "cg")
            print(f"Autoselect preconditioning, niter: {niter}")
            self.assertTrue(niter < 80)



    def test_preconditioned_discriminant_cg(self):
        """Test using preconditioned cg for the discriminant
        classifier."""
        _, offline_data = build_discriminant_traintest_split()
        cpu_mod, gpu_mod = get_discriminant_models("RBF", offline_data,
                num_rffs = NUM_RFFS)

        cpu_mod.set_hyperparams(DISCRIM_HPARAM, offline_data)
        preconditioner, _ = cpu_mod.build_preconditioner(offline_data,
            max_rank = 256, method = "srht")

        niter, _ = cpu_mod.fit(offline_data,  preconditioner = preconditioner,
                max_iter = 500, run_diagnostics = True,
                tol = 1e-6,  mode = "cg")
        print(f"niter: {niter}")
        self.assertTrue(niter < 10)

        if gpu_mod is not None:
            gpu_mod.set_hyperparams(DISCRIM_HPARAM, offline_data)
            preconditioner, _ = gpu_mod.build_preconditioner(offline_data,
                max_rank = 256, method = "srht")

            niter, _ = gpu_mod.fit(offline_data,  preconditioner = preconditioner,
                max_iter = 500, run_diagnostics = True,
                tol = 1e-6,  mode = "cg")
            print(f"Discriminant classifier, niter: {niter}")
            self.assertTrue(niter < 10)


    def test_autoselect_discriminant_cg(self):
        """Test using cg when the software automatically selects the
        max_rank for the discriminant classifier."""
        _, offline_data = build_discriminant_traintest_split()
        cpu_mod, gpu_mod = get_discriminant_models("RBF", offline_data,
                num_rffs = NUM_RFFS)

        cpu_mod.set_hyperparams(DISCRIM_HPARAM, offline_data)
        niter, _ = cpu_mod.fit(offline_data,
                max_iter = 500, run_diagnostics = True,
                tol = 1e-6,  mode = "cg")
        print("Discriminant classifier, autoselected preconditioning, "
                f"niter: {niter}")
        self.assertTrue(niter < 10)

        if gpu_mod is not None:
            gpu_mod.set_hyperparams(DISCRIM_HPARAM, offline_data)
            niter, _ = gpu_mod.fit(offline_data,
                max_iter = 500, run_diagnostics = True,
                tol = 1e-6,  mode = "cg")
            print("Discriminant classifier, autoselected "
                f"preconditioning, niter: {niter}")
            self.assertTrue(niter < 10)



if __name__ == "__main__":
    unittest.main()
