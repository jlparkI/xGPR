"""Checks exact gradients against numerical gradients for
the RBF kernel."""
import unittest
from kernel_specific_gradient_test import run_kernelspecific_test

class CheckRBFGradients(unittest.TestCase):
    """Checks the NMLL gradients for the RBF kernel
    (useful for L-BFGS and SGD hyperparameter tuning)."""

    def test_rbf_gradient(self):
        """Checks that the exact gradient matches numerical."""
        costcomps = run_kernelspecific_test("RBF")
        for costcomp in costcomps:
            self.assertTrue(costcomp)


if __name__ == "__main__":
    unittest.main()
