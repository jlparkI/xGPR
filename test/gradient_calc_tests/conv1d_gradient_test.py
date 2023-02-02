"""Checks exact gradients against numerical gradients for
the Conv1d kernel."""
import unittest
from kernel_specific_gradient_test import run_kernelspecific_test

class CheckConv1dGradients(unittest.TestCase):
    """Checks the NMLL gradients for the Conv1d kernel
    (useful for L-BFGS and SGD hyperparameter tuning)."""

    def test_conv1d_gradient(self):
        costcomps = run_kernelspecific_test("Conv1d",
                        conv_kernel = True)
        for costcomp in costcomps:
            self.assertTrue(costcomp)


if __name__ == "__main__":
    unittest.main()
