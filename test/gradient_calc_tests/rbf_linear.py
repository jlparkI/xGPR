"""Checks exact gradients against numerical gradients for
the RBFLinear kernel."""
import unittest
from kernel_specific_gradient_test import run_kernelspecific_test

class CheckRBFLinearGradients(unittest.TestCase):
    """Checks the NMLL gradients for the GraphRBFLinear kernel
    (useful for L-BFGS and SGD hyperparameter tuning)."""

    def test_rbf_linear_gradient(self):
        """Checks that the exact gradient matches numerical."""
        costcomps = run_kernelspecific_test("RBFPlusLinear",
                        training_rffs = 512, fitting_rffs = 512,
                        conv_kernel = False)
        for costcomp in costcomps:
            self.assertTrue(costcomp)


if __name__ == "__main__":
    unittest.main()
