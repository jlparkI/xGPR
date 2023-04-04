"""Checks exact gradients against numerical gradients for
the GraphRBFLinear kernel."""
import unittest
from kernel_specific_gradient_test import run_kernelspecific_test

class CheckGraphRBFLinearGradients(unittest.TestCase):
    """Checks the NMLL gradients for the GraphRBFLinear kernel
    (useful for L-BFGS and SGD hyperparameter tuning)."""

    def test_graph_rbf_linear_gradient(self):
        """Checks that the exact gradient matches numerical."""
        costcomps = run_kernelspecific_test("GraphRBFPlusLinear",
                        training_rffs = 513, fitting_rffs = 513,
                        conv_kernel = True)
        for costcomp in costcomps:
            self.assertTrue(costcomp)


if __name__ == "__main__":
    unittest.main()
