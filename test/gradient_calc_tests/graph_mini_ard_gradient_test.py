"""Checks exact gradients against numerical gradients for
the GraphMiniARD kernel."""
import unittest
from kernel_specific_gradient_test import run_kernelspecific_test

class CheckGraphMiniARDGradients(unittest.TestCase):
    """Checks the NMLL gradients for the GraphMiniARD kernel."""

    def test_graph_mini_ard_gradient(self):
        costcomps = run_kernelspecific_test("GraphMiniARD",
                        conv_kernel = True,
                        conv_ard_kernel = True)
        for costcomp in costcomps:
            self.assertTrue(costcomp)


if __name__ == "__main__":
    unittest.main()
