"""Tests cg fitting to ensure success both with and without
preconditioning."""
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
HPARAM = np.array([np.log(0.0767),  np.log(0.358)])

NUM_RFFS = 2100
RANDOM_SEED = 123
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
                                512, "cpu")
        print("Preconditioned test with 'easy' hyperparameters for CPU: "
                f"result within {ERROR_MARGIN} percent? {outcome}")
        self.assertTrue(outcome)

        outcome = run_exact_approx_comparison(cpu_mod, HARD_HPARAMS, online_data,
                                512, "cpu")
        print("Preconditioned test with 'hard' hyperparameters for CPU: "
                f"result within {ERROR_MARGIN} percent? {outcome}")
        self.assertTrue(outcome)


        if gpu_mod is not None:
            outcome = run_exact_approx_comparison(gpu_mod, EASY_HPARAMS, online_data,
                                512, "gpu")
            print("Preconditioned test with 'easy' hyperparameters for GPU: "
                f"result within {ERROR_MARGIN} percent? {outcome}")
            self.assertTrue(outcome)

            outcome = run_exact_approx_comparison(gpu_mod, HARD_HPARAMS, online_data,
                                512, "gpu")
            print("Preconditioned test with 'hard' hyperparameters for GPU: "
                f"result within {ERROR_MARGIN} percent? {outcome}")
            self.assertTrue(outcome)


    def test_approximate_nmll(self):
        """Test the approximate nmll function, NO preconditioning."""
        online_data, _ = build_test_dataset(conv_kernel = False)
        cpu_mod, gpu_mod = get_models("RBF", online_data,
                        num_rffs = NUM_RFFS)

        outcome = run_exact_approx_comparison(cpu_mod, EASY_HPARAMS, online_data,
                                0, "cpu")
        print("NO PRECONDITION: Test with 'easy' hyperparameters for CPU: "
                f"result within {ERROR_MARGIN} percent? {outcome}")
        self.assertTrue(outcome)


        if gpu_mod is not None:
            outcome = run_exact_approx_comparison(gpu_mod, EASY_HPARAMS, online_data,
                                0, "gpu")
            print("NO PRECONDITION: Test with 'easy' hyperparameters for GPU: "
                f"result within {ERROR_MARGIN} percent? {outcome}")
            self.assertTrue(outcome)


def run_exact_approx_comparison(model, hyperparams, dataset, max_rank, device):
    """A 'helper' function for generating and comparing exact and
    approximate NMLL."""
    dataset.device = device
    exact_nmll = model.exact_nmll(hyperparams, dataset)
    approx_nmll = model.approximate_nmll(hyperparams, dataset,
                    max_rank = max_rank, nsamples = 25,
                    random_seed = 123, niter = 200, tol = 1e-5)
    outcome = 100 * abs(approx_nmll - exact_nmll) / exact_nmll < ERROR_MARGIN
    print(f"Exact: {exact_nmll}, Approx: {approx_nmll}")
    return outcome


if __name__ == "__main__":
    unittest.main()
