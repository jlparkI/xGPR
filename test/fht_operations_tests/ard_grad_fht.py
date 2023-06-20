"""Tests the feature generation / gradient calculation routines
specific to MiniARD kernels (MiniARD, MiniARDGraph)."""
import sys
import unittest
import numpy as np
from xGPR.kernels.ARD_kernels.mini_ard import MiniARD

try:
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
        outcomes = run_mini_ard_grad_test((10,50), 2000, [25])
        for outcome in outcomes:
            self.assertTrue(outcome)

        outcomes = run_mini_ard_grad_test((10,50), 2000, [25],
                fit_intercept = True)
        for outcome in outcomes:
            self.assertTrue(outcome)

        outcomes = run_mini_ard_grad_test((1000,232), 1000, [100,200])
        for outcome in outcomes:
            self.assertTrue(outcome)

        outcomes = run_mini_ard_grad_test((512,856), 500, [30,450],
                fit_intercept = True)
        for outcome in outcomes:
            self.assertTrue(outcome)





def run_mini_ard_grad_test(xdim, num_freqs, split_points, random_seed = 123,
        fit_intercept = False):
    """A helper function that tests the Cython wrapper which both
    generates RFs and calculates the gradient for specified input params."""
    rng = np.random.default_rng(random_seed)

    input_x = rng.uniform(low=-10.0,high=10.0, size=(xdim[0], xdim[1]))

    gt_double, gt_double_grad = get_MiniARD_gt_values(input_x, num_freqs,
                    split_points, random_seed, double_precision = True,
                    fit_intercept = fit_intercept)

    double_output, double_grad = get_MiniARD_test_values(input_x, num_freqs, split_points,
        random_seed, device = "cpu", double_precision = True,
        fit_intercept = fit_intercept)

    float_output, float_grad = get_MiniARD_test_values(input_x, num_freqs, split_points,
        random_seed, device = "cpu", double_precision = False,
        fit_intercept = fit_intercept)


    if "cupy" in sys.modules:
        cuda_double_output, cuda_double_grad = get_MiniARD_test_values(input_x,
                num_freqs, split_points, random_seed,
                device = "gpu", double_precision = True,
                fit_intercept = fit_intercept)

        cuda_float_output, cuda_float_grad = get_MiniARD_test_values(input_x,
            num_freqs, split_points, random_seed,
            device = "gpu", double_precision = False,
            fit_intercept = fit_intercept)

    outcome_d = np.allclose(gt_double, double_output)
    outcome_f = np.allclose(gt_double, float_output, rtol=1e-5, atol=1e-5)
    outcome_grad_d = np.allclose(gt_double_grad, double_grad)
    outcome_grad_f = np.allclose(gt_double_grad, float_grad, rtol=1e-4, atol=1e-3)
    print("**********\nDid the Grad Calc C extension provide the correct result for RBF of "
            f"{xdim}, {num_freqs}, double precision? {outcome_d}")
    print("**********\nDid the Grad Calc C extension provide the correct result for RBF of "
            f"{xdim}, {num_freqs}, float precision? {outcome_f}")
    print("**********\nDid the Grad Calc C extension provide the correct result for the "
            f"gradient for RBF of {xdim}, {num_freqs}, "
            f"double precision? {outcome_grad_d}")
    print("**********\nDid the Grad Calc C extension provide the correct result for the "
            f"gradient for RBF of {xdim}, {num_freqs}, "
            f"float precision? {outcome_grad_f}")

    if "cupy" in sys.modules:
        outcome_cuda_d = np.allclose(gt_double, cuda_double_output)
        outcome_cuda_f = np.allclose(gt_double, cuda_float_output, rtol=1e-5, atol=1e-5)
        outcome_cuda_grad_d = np.allclose(gt_double_grad, cuda_double_grad)
        outcome_cuda_grad_f = np.allclose(gt_double_grad, cuda_float_grad, rtol=1e-4, atol=1e-3)
        print("**********\nDid the cuda extension provide the correct result for RBF of "
            f"{xdim}, {num_freqs}? {outcome_cuda_d}")
        print("**********\nDid the cuda extension provide the correct result for RBF of "
            f"{xdim}, {num_freqs}? {outcome_cuda_f}")
        print("**********\nDid the Grad Calc cuda extension provide the correct result for the "
            f"gradient for RBF of {xdim}, {num_freqs}? {outcome_cuda_grad_d}")
        print("**********\nDid the Grad Calc cuda extension provide the correct result for the "
            f"gradient for RBF of {xdim}, {num_freqs}? {outcome_cuda_grad_f}")
        return outcome_d, outcome_f, outcome_cuda_d, outcome_cuda_f, \
                outcome_grad_d, outcome_cuda_grad_d
    return outcome_d, outcome_f, outcome_grad_d, outcome_grad_f



def build_mini_ard_kernel(xdim, num_freqs, split_points, random_seed,
        double_precision = False, fit_intercept = False):
    """In this test, we use the MiniARD kernel to do some of
    the legwork in test setup for us. This function builds
    a MiniARD kernel with some specified values."""
    kernel = MiniARD(xdim, 2 * num_freqs, random_seed,
            device = "cpu", double_precision = double_precision,
            kernel_spec_parms = {"split_points":split_points,
                "intercept":fit_intercept})
    kernel.precompute_weights()
    return kernel


def get_MiniARD_gt_values(input_x, num_freqs, split_points,
        random_seed, double_precision = False, fit_intercept = False):
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

    kernel = build_mini_ard_kernel(input_x.shape, num_freqs, split_points,
            random_seed, double_precision = double_precision,
            fit_intercept = fit_intercept)
    no_intercept_kernel = build_mini_ard_kernel(input_x.shape, num_freqs, split_points,
            random_seed, double_precision = double_precision,
            fit_intercept = False)
    xtrans = kernel.transform_x(input_x)
    p_weights = kernel.precomputed_weights.copy()
    #This is an ugly hack that compensates for the fact that the scaling factor
    #is slightly different if using an intercept. TODO: Find a cleaner way to
    #do this in the test.
    if fit_intercept:
        xtrans[:,0] = no_intercept_kernel.transform_x(input_x)[:,0]
        xtrans[:,0] /= np.sqrt(2 / (p_weights.shape[0]))
        xtrans[:,0] *= np.sqrt(2 / (p_weights.shape[0] - 0.5))
    gradient = np.zeros((xtrans.shape[0], xtrans.shape[1],
                    len(split_points) + 1))

    ks = kernel.split_pts
    for i in range(ks.shape[0] - 1):
        temp_arr = np.einsum("ij,kj->ki", p_weights[:, ks[i]:ks[i+1]],
                                    retyped_x[:,ks[i]:ks[i+1]])
        for j in range(p_weights.shape[0]):
            gradient[:,2*j,i] = -temp_arr[:,j] * xtrans[:,2*j+1]
            gradient[:,2*j+1,i] = temp_arr[:,j] * xtrans[:,2*j]
    if fit_intercept:
        gradient[:,0,:] = 0
        xtrans[:,0] = kernel.hyperparams[1]
    return xtrans, gradient


def get_MiniARD_test_values(input_x, num_freqs, split_points,
        random_seed, device = "cpu", double_precision = False,
        fit_intercept = False):
    """Generates the test values for comparison with the ground-
    truth values. In this case, we use the MiniARD kernel for
    convenience."""
    kernel = build_mini_ard_kernel(input_x.shape, num_freqs, split_points,
            random_seed, double_precision = double_precision,
            fit_intercept = fit_intercept)
    kernel.device = device
    if device == "gpu":
        x_device = cp.asarray(input_x)
    else:
        x_device = input_x
    xtrans, gradient = kernel.kernel_specific_gradient(x_device)
    if device == "gpu":
        xtrans = cp.asnumpy(xtrans)
        gradient = cp.asnumpy(gradient)
    return xtrans, gradient



if __name__ == "__main__":
    unittest.main()
