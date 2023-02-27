"""Tests using both exact fitting and preconditioned CG with
minimal bayes tuning to ensure that we can achieve expected
performance with the Conv1d kernel. This is an 'all-in-one'
workflow test, if it fails, run fitting tests, tuning tests,
preconditioner tests and fht operations tests as appopriate
to determine which component is failing."""
import unittest

from test_fitting_utils import test_fit_cpu, test_fit_gpu

RANDOM_SEED = 123
CONV_KERNEL = True
KERNEL = "Conv1d"


class CheckConv1dPipeline(unittest.TestCase):
    """An all in one pipeline test."""

    def test_fit_cpu(self):
        """Test on CPU using preconditioned CG and exact."""
        cg_score, exact_score = test_fit_cpu(KERNEL, CONV_KERNEL, RANDOM_SEED,
                conv_width = 3)
        self.assertTrue(cg_score > 0.59)
        self.assertTrue(exact_score > 0.59)


    def test_fit_gpu(self):
        """Test on GPU using preconditioned CG and exact."""
        cg_score, exact_score = test_fit_gpu(KERNEL, CONV_KERNEL, RANDOM_SEED,
                conv_width = 3)
        self.assertTrue(cg_score > 0.59)
        self.assertTrue(exact_score > 0.59)


if __name__ == "__main__":
    unittest.main()
