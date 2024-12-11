"""Tests cg fitting to ensure success both with and without
preconditioning."""
import sys
import os
import unittest

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.build_test_dataset import build_test_dataset
from utils.model_constructor import get_models
from utils.evaluate_model import evaluate_model

#A set of hyperparameters known to work well for our testing dataset
#that we can use as a default.
HPARAM = np.array([np.log(0.0767),  np.log(0.358)])

NUM_RFFS = 2100
ERROR_MARGIN = 1.0
EASY_HPARAMS = np.array([0.,  1.0])
HARD_HPARAMS = np.array([np.log(1e-3),  1.0])

class CheckApproximateNMLL(unittest.TestCase):
    """Checks whether the approximate NMLL is a reasonably
    good approximation to the exact NMLL."""

    def test_approximate_preconditioned_nmll(self):
        """Test the approximate nmll function using preconditioning."""
        online_data, _ = build_test_dataset(conv_kernel = False)
        cpu_mod, gpu_mod = get_models("RBF", online_data, num_rffs = NUM_RFFS)

        outcome = run_exact_approx_comparison(cpu_mod, EASY_HPARAMS, online_data,
                                "cpu")
        print("Preconditioned test with 'easy' hyperparameters for CPU: "
                f"result within {ERROR_MARGIN} percent? {outcome}")
        self.assertTrue(outcome)

        outcome = run_exact_approx_comparison(cpu_mod, HARD_HPARAMS, online_data,
                                "cpu")
        print("Preconditioned test with 'hard' hyperparameters for CPU: "
                f"result within {ERROR_MARGIN} percent? {outcome}")
        self.assertTrue(outcome)


        if gpu_mod is not None:
            outcome = run_exact_approx_comparison(gpu_mod, EASY_HPARAMS, online_data,
                                "gpu")
            print("Preconditioned test with 'easy' hyperparameters for GPU: "
                f"result within {ERROR_MARGIN} percent? {outcome}")
            self.assertTrue(outcome)

            outcome = run_exact_approx_comparison(gpu_mod, HARD_HPARAMS, online_data,
                                "gpu")
            print("Preconditioned test with 'hard' hyperparameters for GPU: "
                f"result within {ERROR_MARGIN} percent? {outcome}")
            self.assertTrue(outcome)


    def test_approximate_nmll(self):
        """Test the approximate nmll function, NO preconditioning."""
        online_data, _ = build_test_dataset(conv_kernel = False)
        cpu_mod, gpu_mod = get_models("RBF", online_data,
                        num_rffs = NUM_RFFS)



def run_exact_approx_comparison(model, hyperparams, dataset, device):
    """A 'helper' function for generating and comparing exact and
    approximate NMLL."""
    exact_nmll = model.exact_nmll(hyperparams, dataset)
    approx_nmll = model.approximate_nmll(hyperparams, dataset)
    outcome = 100 * abs(approx_nmll - exact_nmll) / exact_nmll < ERROR_MARGIN
    print(f"Exact: {exact_nmll}, Approx: {approx_nmll}")
    return outcome


if __name__ == "__main__":
    unittest.main()
