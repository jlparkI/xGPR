"""Tests using both exact fitting and preconditioned CG with
minimal bayes tuning to ensure that we can achieve expected
performance with the Conv1d kernel. This is an 'all-in-one'
workflow test, if it fails, run fitting tests, tuning tests,
preconditioner tests and fht operations tests as appopriate
to determine which component is failing."""
import unittest

from test_fitting_utils import test_fit_cg, test_fit_exact

RANDOM_SEED = 123
CONV_KERNEL = True
KERNEL = "Conv1d"


class CheckConv1dPipeline(unittest.TestCase):
    """An all in one pipeline test."""

    def test_fit_cg(self):
        """Test using preconditioned cg."""
        cpu_score, gpu_score = test_fit_cg(KERNEL, CONV_KERNEL, RANDOM_SEED,
                conv_width = 3)
        self.assertTrue(cpu_score > 0.61)
        if gpu_score is not None:
            self.assertTrue(gpu_score > 0.61)


    def test_fit_exact(self):
        """Test using exact."""
        cpu_score, gpu_score = test_fit_exact(KERNEL, CONV_KERNEL, RANDOM_SEED,
                conv_width = 3)
        self.assertTrue(cpu_score > 0.62)
        if gpu_score is not None:
            self.assertTrue(gpu_score > 0.62)


if __name__ == "__main__":
    unittest.main()
