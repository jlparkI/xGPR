"""Tests the RBF feature generation routines (specific for RBF, Matern
and MiniARD / ARD kernels, which by extension includes static layer kernels."""
import sys
import unittest
from math import ceil
import numpy as np
from scipy.stats import chi
import cupy as cp

from xGPR.xgpr_cpu_rfgen_cpp_ext import cpuRBFFeatureGen as cRBF
from xGPR.xgpr_cpu_rfgen_cpp_ext import cpuRBFGrad as cRBFGrad
from xGPR.xgpr_cpu_rfgen_cpp_ext import cpuFastHadamardTransform2D as cFHT

from xGPR.xgpr_cuda_rfgen_cpp_ext import cudaRBFFeatureGen as cudaRBF
from xGPR.xgpr_cuda_rfgen_cpp_ext import cudaRBFGrad


class TestRBFFeatureGen(unittest.TestCase):
    """Runs tests for RBF feature generation for float and double
    precision for CPU and (if available) GPU. We do this by comparing
    with feature generation using a Python routine that
    makes use of SORFTransform functions, so the basic_fht test must
    pass in order for this test to work."""


    def test_rbf_feature_gen(self):
        """Tests RBF feature generation for CPU and if available GPU."""
        for n_freqs in [64, 2000, 8192]:
            outcomes = run_rbf_test((10,50), n_freqs)
            for outcome in outcomes:
                self.assertTrue(outcome)

            outcomes = run_rbf_test((10,3), n_freqs)
            for outcome in outcomes:
                self.assertTrue(outcome)

            outcomes = run_rbf_test((3,2003), n_freqs)
            for outcome in outcomes:
                self.assertTrue(outcome)

            outcomes = run_rbf_test((11,1076), n_freqs)
            for outcome in outcomes:
                self.assertTrue(outcome)

            outcomes = run_rbf_test((231,856), n_freqs, fit_intercept = True)
            for outcome in outcomes:
                self.assertTrue(outcome)



    def test_rbf_grad_calc(self):
        """Tests RBF gradient calc for CPU and if available GPU."""
        for n_freqs in [64, 2000, 8192]:
            outcomes = run_rbf_grad_test((10,50), n_freqs)
            for outcome in outcomes:
                self.assertTrue(outcome)

            outcomes = run_rbf_grad_test((513,232), n_freqs, fit_intercept = True)
            for outcome in outcomes:
                self.assertTrue(outcome)

            outcomes = run_rbf_grad_test((11,3001), n_freqs)
            for outcome in outcomes:
                self.assertTrue(outcome)

            outcomes = run_rbf_grad_test((11,3), n_freqs)
            for outcome in outcomes:
                self.assertTrue(outcome)


def run_rbf_test(xdim, num_freqs, random_seed = 123, fit_intercept = False):
    """A helper function that runs the RBF test for
    specified input dimensions."""

    test_array, radem, chi_arr, nblocks, padded_dims = \
            setup_rbf_test(xdim, num_freqs, random_seed)

    gt_double, _ = generate_rbf_values(test_array, radem, chi_arr,
            nblocks, padded_dims, fit_intercept)
    gt_float, _ = generate_rbf_values(test_array.astype(np.float32), radem,
            chi_arr.astype(np.float32), nblocks, padded_dims, fit_intercept)

    double_output = np.zeros((test_array.shape[0], num_freqs * 2))
    cRBF(test_array, double_output, radem, chi_arr, 2, fit_intercept)

    float_output = np.zeros((test_array.shape[0], num_freqs * 2))
    cRBF(test_array.astype(np.float32), float_output, radem,
            chi_arr.astype(np.float32), 2, fit_intercept)

    if "cupy" in sys.modules:
        cuda_test_array = cp.asarray(test_array)
        radem = cp.asarray(radem)
        chi_arr = cp.asarray(chi_arr)
        cuda_double_output = cp.zeros((test_array.shape[0], num_freqs * 2))
        cuda_float_output = cp.zeros((test_array.shape[0], num_freqs * 2))

        cudaRBF(cuda_test_array, cuda_double_output, radem,
                chi_arr, fit_intercept)
        cudaRBF(cuda_test_array.astype(cp.float32), cuda_float_output, radem,
                chi_arr.astype(cp.float32), fit_intercept)


    outcome_d = np.allclose(gt_double, double_output)
    outcome_f = np.allclose(gt_float, float_output, rtol=1e-5, atol=1e-4)
    print("Correct result for CPU  for RBF of "
            f"{xdim}, {num_freqs} for float, double? {outcome_f},{outcome_d}")

    if "cupy" in sys.modules:
        outcome_cuda_d = np.allclose(gt_double, cuda_double_output)
        outcome_cuda_f = np.allclose(gt_float, cuda_float_output, rtol=1e-5,
                atol=1e-4)
        print("Correct result for Cuda for RBF of "
            f"{xdim}, {num_freqs} for float, double? {outcome_cuda_f},{outcome_cuda_d}")
        return outcome_d, outcome_f, outcome_cuda_d, outcome_cuda_f
    return outcome_d, outcome_f




def run_rbf_grad_test(xdim, num_freqs, random_seed = 123,
        fit_intercept = False):
    """A helper function that tests the Cython wrapper which both
    generates RFs and calculates the gradient for specified input params."""

    test_array, radem, chi_arr, nblocks, padded_dims = \
            setup_rbf_test(xdim, num_freqs, random_seed)

    gt_double, gt_double_grad = generate_rbf_values(test_array, radem, chi_arr,
            nblocks, padded_dims, fit_intercept)
    gt_float, gt_float_grad = generate_rbf_values(test_array.astype(np.float32),
            radem, chi_arr.astype(np.float32), nblocks, padded_dims,
            fit_intercept)

    double_output = np.zeros((test_array.shape[0], num_freqs * 2))
    double_grad = np.zeros((double_output.shape[0], double_output.shape[1], 1))
    cRBFGrad(test_array, double_output, double_grad, radem, chi_arr,
            1.0, 2, fit_intercept)

    float_output = np.zeros((test_array.shape[0], num_freqs * 2))
    float_grad = np.zeros((float_output.shape[0], float_output.shape[1], 1))
    cRBFGrad(test_array.astype(np.float32), float_output,
            float_grad, radem, chi_arr.astype(np.float32), 1.0,
            2, fit_intercept)

    if "cupy" in sys.modules:
        cuda_test_array = cp.asarray(test_array)
        radem = cp.asarray(radem)
        chi_arr = cp.asarray(chi_arr)
        cuda_double_output = cp.zeros((test_array.shape[0], num_freqs * 2))
        cuda_float_output = cp.zeros((test_array.shape[0], num_freqs * 2))
        cuda_double_grad = cp.zeros((test_array.shape[0], num_freqs * 2, 1))
        cuda_float_grad = cp.zeros((test_array.shape[0], num_freqs * 2, 1))

        cudaRBFGrad(cuda_test_array, cuda_double_output, cuda_double_grad, radem,
                chi_arr, 1.0, fit_intercept)
        cudaRBFGrad(cuda_test_array.astype(cp.float32),
                cuda_float_output, cuda_float_grad, radem,
                chi_arr.astype(cp.float32), 1.0,
                fit_intercept)

    outcome_d = np.allclose(gt_double, double_output)
    outcome_f = np.allclose(gt_float, float_output, atol=1e-4,
            rtol=1e-4)
    outcome_grad_d = np.allclose(gt_double_grad, double_grad)
    outcome_grad_f = np.allclose(gt_float_grad, float_grad,
            atol=1e-4, rtol=1e-4)
    print("Correct result for CPU  for gradient RBF of "
            f"{xdim}, {num_freqs} float, double? {outcome_f},{outcome_d}")
    print("Correct result for Cuda for gradient RBF of "
            f"{xdim}, {num_freqs} float, double? "
            f"{outcome_grad_f},{outcome_grad_d}")

    if "cupy" in sys.modules:
        outcome_cuda_d = np.allclose(gt_double, cuda_double_output)
        outcome_cuda_f = np.allclose(gt_float, cuda_float_output,
                atol=1e-4, rtol=1e-4)
        outcome_cuda_grad_d = np.allclose(gt_double_grad, cuda_double_grad)
        outcome_cuda_grad_f = np.allclose(gt_float_grad, cuda_float_grad,
                atol=1e-2, rtol=1e-2)
        print("Correct result for CPU  for gradient RBF of "
            f"{xdim}, {num_freqs} float, double? {outcome_cuda_f},{outcome_cuda_d}")
        print("Correct result for Cuda for gradient RBF of "
            f"{xdim}, {num_freqs} float, double? "
            f"{outcome_cuda_grad_f},{outcome_cuda_grad_d}")
        return outcome_d, outcome_f, outcome_cuda_d, outcome_cuda_f, \
                outcome_grad_d, outcome_grad_f, outcome_cuda_grad_d, \
                outcome_cuda_grad_f
    return outcome_d, outcome_f, outcome_grad_d, outcome_grad_f



def setup_rbf_test(xdim, num_freqs, random_seed = 123):
    """A helper function that builds the matrices required for
    the RBF test, specified using the input dimensions."""
    padded_dims = 2**ceil(np.log2(max(xdim[-1], 2)))
    if padded_dims < num_freqs:
        nblocks = ceil(num_freqs / padded_dims)
    else:
        nblocks = 1

    radem_array = np.asarray([-1,1], dtype=np.int8)
    rng = np.random.default_rng(random_seed)


    radem = rng.choice(radem_array, size=(3,1,nblocks * padded_dims),
            replace=True)
    chi_arr = chi.rvs(df=padded_dims, size=num_freqs,
                            random_state = random_seed)

    test_array = rng.uniform(low=-10.0,high=10.0, size=(xdim[0], xdim[1]))
    return test_array, radem, chi_arr, nblocks, padded_dims


def generate_rbf_values(test_array, radem, chi_arr, nblocks,
        padded_dims, fit_intercept = False):
    """Generates the 'ground-truth' RBF values for
    comparison with those from the C / Cuda extension,
    using a very inefficient but easy to troubleshoot
    pure Python approach."""
    pretrans_x = []
    norm_constant = np.log2(padded_dims) / 2
    norm_constant = 1 / (2**norm_constant)

    for i in range(nblocks):
        temp_arr = np.zeros((test_array.shape[0], padded_dims),
                dtype = test_array.dtype)
        temp_arr[:,:test_array.shape[1]] = test_array

        for j in range(3):
            temp_arr *= radem[j, 0, i*padded_dims:(i+1)*padded_dims] * norm_constant
            cFHT(temp_arr, 1)

        #The simplex projection (Reid et al 2023) (disabled for now
        #since erratic / unclear effect on performance). We use a deliberately
        #clumsy / inefficient approach here to ensure that numpy
        #will use 32-bit float precision if the input is 32-bit
        #(otherwise numpy tends to default to 64-bit and the result
        #may not be np.allclose to the 32-bit c extension calculation)
        #scalar = np.sqrt(padded_dims - 1, dtype=temp_arr.dtype)
        #sum_arr = np.zeros((temp_arr.shape[0]), dtype=temp_arr.dtype)
        #for j in range(temp_arr.shape[1] - 1):
        #    sum_arr += temp_arr[:,j]
        #sum_arr /= scalar
        #temp_arr[:,-1] = sum_arr
        #scalar = ((1 + np.sqrt(padded_dims, dtype=temp_arr.dtype)) /
        #        (padded_dims - 1)).astype(temp_arr.dtype)
        #sum_arr *= scalar
        #scalar = np.sqrt(padded_dims / (padded_dims - 1),
        #        dtype=temp_arr.dtype)
        #temp_arr[:,:-1] = temp_arr[:,:-1] * scalar - sum_arr[:,None]

        pretrans_x.append(temp_arr)

    pretrans_x = np.hstack(pretrans_x)[:,:chi_arr.shape[0]]
    pretrans_x *= chi_arr[None,:]


    xtrans = np.zeros((test_array.shape[0], chi_arr.shape[0] * 2))
    gradient = np.zeros((test_array.shape[0], chi_arr.shape[0] * 2, 1))

    for j in range(0, chi_arr.shape[0], 1):
        xtrans[:,2*j] = np.cos(pretrans_x[:,j])
        xtrans[:,2*j+1] = np.sin(pretrans_x[:,j])

        gradient[:,2*j,0] = -xtrans[:,2*j+1] * \
                pretrans_x[:,j].astype(np.float64)
        gradient[:,2*j+1,0] = xtrans[:,2*j] * \
                pretrans_x[:,j].astype(np.float64)

    if fit_intercept:
        xtrans *= np.sqrt(1 / (chi_arr.shape[0]-0.5))
        gradient *= np.sqrt(1 / (chi_arr.shape[0]-0.5))
    else:
        xtrans *= np.sqrt(1 / chi_arr.shape[0])
        gradient *= np.sqrt(1 / chi_arr.shape[0])
    return xtrans, gradient


if __name__ == "__main__":
    unittest.main()
