"""Tests the feature generation / gradient calculation routines
specific to MiniARD kernels (MiniARD, MiniARDGraph)."""
import sys
import unittest
import timeit
from math import ceil
import numpy as np
from xGPR.kernels.ARD_kernels.mini_ard import MiniARD
from cpu_rbf_operations import doubleCpuMiniARDGrad as dMiniARDGrad
from cpu_rbf_operations import floatCpuRBFGrad as fMiniARDGrad

from cpu_basic_operations import doubleCpuSORFTransform as dSORF
from cpu_basic_operations import floatCpuSORFTransform as fSORF

try:
    from cuda_rbf_operations import doubleCudaMiniARDGrad as dCudaMiniARDGrad
    from cuda_rbf_operations import floatCudaMiniARDGrad as fCudaMiniARDGrad
    import cupy as cp
except:
    pass


class TestARDGradCalcs(unittest.TestCase):
    """Runs tests for wrapped ARD gradient calc functions for float and double
    precision for CPU and (if available) GPU. We do this by comparing
    with feature generation using a Python routine that
    makes use of SORFTransform functions, so the basic_fht test must
    pass in order for this test to work."""


    def test_mini_ard_calc(self):
        """Tests gradient calculation for the MiniARD kernel."""
        outcomes = run_grad_test((10,50), 2000, [25])
        for outcome in outcomes:
            self.assertTrue(outcome)

        outcomes = run_grad_test((1000,232), 1000, [100,200])
        for outcome in outcomes:
            self.assertTrue(outcome)

        outcomes = run_grad_test((512,856), 500, [30,450])
        for outcome in outcomes:
            self.assertTrue(outcome)






def run_grad_test(xdim, num_freqs, split_points, random_seed = 123):
    """A helper function that tests the Cython wrapper which both
    generates RFs and calculates the gradient for specified input params."""
    rng = np.random.default_rng(random_seed)

    input_x = rng.uniform(low=-10.0,high=10.0, size=(xdim[0], xdim[1]))

    gt_double, gt_double_grad = get_MiniARD_gt_values(input_x, num_freqs,
                    split_points, random_seed, double_precision = True)
    gt_float, gt_float_grad = get_MiniARD_gt_values(input_x, num_freqs,
                    split_points, random_seed, double_precision = False)

    double_output, double_grad = get_MiniARD_test_values(input_x, num_freqs, split_points,
        random_seed, device = "cpu", double_precision = True)

    float_output, float_grad = get_MiniARD_test_values(input_x, num_freqs, split_points,
        random_seed, device = "cpu", double_precision = False)


    if "cupy" in sys.modules:
        cuda_double_output, cuda_double_grad = get_MiniARD_test_values(input_x,
                num_freqs, split_points, random_seed,
                device = "gpu", double_precision = True)

        cuda_float_output, cuda_float_grad = get_MiniARD_test_values(input_x,
            num_freqs, split_points, random_seed,
            device = "gpu", double_precision = False)

    outcome_d = np.allclose(gt_double, double_output)
    outcome_f = np.allclose(gt_float, float_output, rtol=1e-6, atol=1e-6)
    outcome_grad_d = np.allclose(gt_double_grad, double_grad)
    outcome_grad_f = np.allclose(gt_float_grad, float_grad, rtol=1e-6, atol=1e-6)
    print("**********\nDid the Grad Calc C extension provide the correct result for RBF of "
            f"{xdim}, {num_freqs}? {outcome_d}\n*******")
    print("**********\nDid the Grad Calc C extension provide the correct result for RBF of "
            f"{xdim}, {num_freqs}? {outcome_f}\n*******")
    print("**********\nDid the Grad Calc C extension provide the correct result for the "
            f"gradient for RBF of {xdim}, {num_freqs}? {outcome_grad_d}\n*******")
    print("**********\nDid the Grad Calc C extension provide the correct result for the "
            f"gradient for RBF of {xdim}, {num_freqs}? {outcome_grad_f}\n*******")

    if "cupy" in sys.modules:
        outcome_cuda_d = np.allclose(gt_double, cuda_double_output)
        outcome_cuda_f = np.allclose(gt_float, cuda_float_output, rtol=1e-6, atol=1e-6)
        outcome_cuda_grad_d = np.allclose(gt_double_grad, cuda_double_grad)
        outcome_cuda_grad_f = np.allclose(gt_float_grad, cuda_float_grad, rtol=1e-6, atol=1e-6)
        print("**********\nDid the cuda extension provide the correct result for RBF of "
            f"{xdim}, {num_freqs}? {outcome_cuda_d}\n*******")
        print("**********\nDid the cuda extension provide the correct result for RBF of "
            f"{xdim}, {num_freqs}? {outcome_cuda_f}\n*******")
        print("**********\nDid the Grad Calc cuda extension provide the correct result for the "
            f"gradient for RBF of {xdim}, {num_freqs}? {outcome_cuda_grad_d}\n*******")
        print("**********\nDid the Grad Calc cuda extension provide the correct result for the "
            f"gradient for RBF of {xdim}, {num_freqs}? {outcome_cuda_grad_f}\n*******")
        return outcome_d, outcome_f, outcome_cuda_d, outcome_cuda_f, \
                outcome_grad_d, outcome_grad_f, outcome_cuda_grad_d, \
                outcome_cuda_grad_f
    return outcome_d, outcome_f, outcome_grad_d, outcome_grad_f



def build_kernel(xdim, num_freqs, split_points, random_seed,
        double_precision = False):
    """In this test, we use the MiniARD kernel to do some of
    the legwork in test setup for us. This function builds
    a MiniARD kernel with some specified values."""
    kernel = MiniARD(xdim, 2 * num_freqs, random_seed,
            device = "cpu", double_precision = double_precision,
            kernel_spec_parms = {"split_points":split_points})
    kernel.precompute_weights()
    return kernel



def get_MiniARD_gt_values(input_x, num_freqs, split_points,
        random_seed, double_precision = False):
    """Generates the 'ground-truth' MiniARD grad values for
    comparison with those from the C / Cuda extension. Since
    the MiniARD kernel uses the rbfFeatureGen wrapped functions
    to generate random features when transform_x is called,
    we use these as 'ground truth' for rfs and compare them
    with those generated by the kernel_specific_gradient function.
    We then calculate the 'ground truth' gradient separately."""
    if double_precision:
        retyped_x = input_x.astype(np.float64)
    else:
        retyped_x = input_x.astype(np.float32)

    kernel = build_kernel(input_x.shape, num_freqs, split_points,
            random_seed, double_precision = double_precision)
    xtrans = kernel.transform_x(input_x)
    p_weights = kernel.precomputed_weights.copy()
    gradient = np.zeros((xtrans.shape[0], xtrans.shape[1],
                    len(split_points) + 1))

    for i in range(len(split_points) + 1):
        gradient[:,:num_freqs,i] = np.einsum("ij,kj->ki", p_weights, retyped_x)
        gradient[:,:num_freqs,i] *= kernel.hyperparams[2+1]
        gradient[:,num_freqs:,i] = gradient[:,:num_freqs,i] * xtrans[:,:num_freqs]
        gradient[:,:num_freqs,i] *= -xtrans[:,num_freqs:]
    return xtrans, gradient


def get_MiniARD_test_values(input_x, num_freqs, split_points,
        random_seed, device = "cpu", double_precision = False):
    """Generates the test values for comparison with the ground-
    truth values. In this case, we use the MiniARD kernel for
    convenience."""
    kernel = build_kernel(input_x.shape, num_freqs, split_points,
            random_seed, double_precision = double_precision)
    kernel.device = device
    if device == "gpu":
        x_device = cp.asarray(input_x)
    else:
        x_device = input_x
    xtrans, gradient = kernel.kernel_specific_gradient(x_device)
    return xtrans, gradient


if __name__ == "__main__":
    unittest.main()
