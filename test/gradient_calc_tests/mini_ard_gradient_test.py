"""Checks exact gradients against numerical gradients for
the MiniARD kernel."""
import unittest
from kernel_specific_gradient_test import run_kernelspecific_test

class CheckMiniARDGradients(unittest.TestCase):
    """Checks the NMLL gradients for the MiniARD kernel."""

    def test_mini_ard_gradient(self):
        costcomps = run_kernelspecific_test("MiniARD",
                        conv_kernel = False)
        for costcomp in costcomps:
            self.assertTrue(costcomp)


if __name__ == "__main__":
    unittest.main()
