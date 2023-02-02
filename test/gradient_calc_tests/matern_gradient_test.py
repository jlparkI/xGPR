"""Checks exact gradients against numerical gradients for
the Matern kernel."""
import unittest
from kernel_specific_gradient_test import run_kernelspecific_test

class CheckMaternGradients(unittest.TestCase):
    """Checks the NMLL gradients for the Matern kernel
    (useful for L-BFGS and SGD hyperparameter tuning)."""

    def test_matern_gradient(self):
        """Checks that the exact gradient matches numerical."""
        costcomps = run_kernelspecific_test("Matern")
        for costcomp in costcomps:
            self.assertTrue(costcomp)


if __name__ == "__main__":
    unittest.main()
