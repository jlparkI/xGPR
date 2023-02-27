"""Tests using both exact fitting and preconditioned CG with
minimal bayes tuning to ensure that we can achieve expected
performance with the Classic Polynomial kernel. This is an
'all-in-one' workflow test, if it fails, run fitting tests,
tuning tests, preconditioner tests and fht operations tests
as appopriate to determine which component is failing."""
import unittest

from test_fitting_utils import test_fit_cpu, test_fit_gpu

RANDOM_SEED = 123
CONV_KERNEL = False
KERNEL = "Poly"


class CheckClassicPolyPipeline(unittest.TestCase):
    """An all in one pipeline test."""


    def test_fit_cpu(self):
        """Test on cpu."""
        cg_score, exact_score = test_fit_cpu(KERNEL, CONV_KERNEL, RANDOM_SEED,
                conv_width = 3)
        self.assertTrue(cg_score > 0.38)
        self.assertTrue(exact_score > 0.38)

    def test_fit_gpu(self):
        """Test on gpu."""
        cg_score, exact_score = test_fit_gpu(KERNEL, CONV_KERNEL, RANDOM_SEED,
                conv_width = 3)
        if cg_score is None or exact_score is None:
            return
        self.assertTrue(cg_score > 0.38)
        self.assertTrue(exact_score > 0.38)


if __name__ == "__main__":
    unittest.main()
