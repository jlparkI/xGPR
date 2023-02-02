"""Checks exact gradients against numerical gradients for
the Linear kernel."""
import unittest
from kernel_specific_gradient_test import run_kernelspecific_test

class CheckLinearGradients(unittest.TestCase):
    """Checks the NMLL gradients for the Linear kernel
    (useful for L-BFGS and SGD hyperparameter tuning)."""

    def test_linear_gradient(self):
        """Checks that the exact gradient matches numerical."""
        costcomps = run_kernelspecific_test("Linear")
        for costcomp in costcomps:
            self.assertTrue(costcomp)


if __name__ == "__main__":
    unittest.main()
