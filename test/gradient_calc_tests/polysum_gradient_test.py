"""Checks exact gradients against numerical gradients for
the PolySum kernel."""
import unittest
from kernel_specific_gradient_test import run_kernelspecific_test

class CheckPolySumGradients(unittest.TestCase):
    """Checks the NMLL gradients for the PolySum kernel
    (useful for L-BFGS and SGD hyperparameter tuning)."""

    def test_conv1d_gradient(self):
        """Checks that the exact gradient matches numerical."""
        costcomps = run_kernelspecific_test("GraphPoly",
                        conv_kernel = True)
        for costcomp in costcomps:
            self.assertTrue(costcomp)


if __name__ == "__main__":
    unittest.main()
