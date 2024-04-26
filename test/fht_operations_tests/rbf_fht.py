"""Tests the RBF feature generation routines (specific for RBF, Matern
and MiniARD / ARD kernels, which by extension includes static layer kernels."""
import unittest
from math import ceil
import numpy as np
from scipy.stats import chi
from cpu_rf_gen_module import cpuRBFFeatureGen as cRBF
from cpu_rf_gen_module import cpuRBFGrad as cRBFGrad

from cpu_rf_gen_module import cpuFastHadamardTransform2D as cFHT

from cuda_rf_gen_module import cudaRBFFeatureGen as cudaRBF
from cuda_rf_gen_module import cudaRBFGrad as cudaRBFGrad


class TestRBFFeatureGen(unittest.TestCase):
    """Runs tests for RBF feature generation for float and double
    precision for CPU and (if available) GPU. We do this by comparing
    with feature generation using a Python routine that
    makes use of SORFTransform functions, so the basic_fht test must
    pass in order for this test to work."""


    def test_rbf_feature_gen(self):
        """Tests RBF feature generation for CPU and if available GPU."""
        outcomes = run_rbf_test((10,50), 2000)
        for outcome in outcomes:
            self.assertTrue(outcome)

        outcomes = run_rbf_test((1000,232), 1000)
        for outcome in outcomes:
            self.assertTrue(outcome)

        outcomes = run_rbf_test((512,856), 500, fit_intercept = True)
        for outcome in outcomes:
            self.assertTrue(outcome)

    def test_rbf_grad_calc(self):
        """Tests RBF gradient calc for CPU and if available GPU."""
        outcomes = run_rbf_grad_test((10,50), 2000)
        for outcome in outcomes:
            self.assertTrue(outcome)

        outcomes = run_rbf_grad_test((1000,232), 1000, fit_intercept = True)
        for outcome in outcomes:
            self.assertTrue(outcome)

        outcomes = run_rbf_grad_test((512,856), 500)
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

    '''if "cupy" in sys.modules:
        cuda_test_array = cp.asarray(test_array)
        cuda_test_float = cp.asarray(test_float)
        radem = cp.asarray(radem)
        chi_arr = cp.asarray(chi_arr)
        chi_float = cp.asarray(chi_float)
        cuda_double_output = cp.zeros((test_array.shape[0], num_freqs * 2))
        cuda_float_output = cp.zeros((test_array.shape[0], num_freqs * 2))

        cudaRBF(cuda_test_array, cuda_double_output, radem,
                chi_arr, 2, fit_intercept)
        cudaRBF(cuda_test_float, cuda_float_output, radem,
                chi_float, 2, fit_intercept)'''


    outcome_d = np.allclose(gt_double, double_output)
    outcome_f = np.allclose(gt_float, float_output)
    print("**********\nDid the C extension provide the correct result for RBF of "
            f"{xdim}, {num_freqs}? {outcome_d}")
    print("**********\nDid the C extension provide the correct result for RBF of "
            f"{xdim}, {num_freqs}? {outcome_f}")

    '''if "cupy" in sys.modules:
        outcome_cuda_d = np.allclose(gt_double, cuda_double_output)
        outcome_cuda_f = np.allclose(gt_float, cuda_float_output)
        print("**********\nDid the cuda extension provide the correct result for RBF of "
            f"{xdim}, {num_freqs}? {outcome_cuda_d}")
        print("**********\nDid the cuda extension provide the correct result for RBF of "
            f"{xdim}, {num_freqs}? {outcome_cuda_f}")
        return outcome_d, outcome_f, outcome_cuda_d, outcome_cuda_f'''
    return outcome_d, outcome_f




def run_rbf_grad_test(xdim, num_freqs, random_seed = 123, fit_intercept = False):
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
    double_grad = cRBFGrad(test_array, double_output, radem, chi_arr,
            sigmaHparam = 1.0, numThreads = 2, fitIntercept = fit_intercept)

    float_output = np.zeros((test_array.shape[0], num_freqs * 2))
    float_grad = cRBFGrad(test_array.astype(np.float32), float_output,
            radem, chi_arr.astype(np.float32), sigmaHparam = 1.0,
            numThreads = 2, fitIntercept = fit_intercept)

    '''if "cupy" in sys.modules:
        cuda_test_array = cp.asarray(test_array)
        cuda_test_float = cp.asarray(test_float)
        radem = cp.asarray(radem)
        chi_arr = cp.asarray(chi_arr)
        chi_float = cp.asarray(chi_float)
        cuda_double_output = cp.zeros((test_array.shape[0], num_freqs * 2))
        cuda_float_output = cp.zeros((test_array.shape[0], num_freqs * 2))

        cuda_double_grad = cudaRBFGrad(cuda_test_array, cuda_double_output, radem,
                chi_arr, sigmaHparam = 1.0, numThreads = 2,
                fitIntercept = fit_intercept)
        cuda_float_grad = cudaRBFGrad(cuda_test_float, cuda_float_output, radem,
                chi_float, sigmaHparam = 1.0, numThreads = 2,
                fitIntercept = fit_intercept)'''

    outcome_d = np.allclose(gt_double, double_output)
    outcome_f = np.allclose(gt_float, float_output)
    outcome_grad_d = np.allclose(gt_double_grad, double_grad)
    outcome_grad_f = np.allclose(gt_float_grad, float_grad)
    print("**********\nDid the Grad Calc C extension provide the correct result for RBF of "
            f"{xdim}, {num_freqs}? {outcome_d}")
    print("**********\nDid the Grad Calc C extension provide the correct result for RBF of "
            f"{xdim}, {num_freqs}? {outcome_f}")
    print("**********\nDid the Grad Calc C extension provide the correct result for the "
            f"gradient for RBF of {xdim}, {num_freqs}? {outcome_grad_d}")
    print("**********\nDid the Grad Calc C extension provide the correct result for the "
            f"gradient for RBF of {xdim}, {num_freqs}? {outcome_grad_f}")

    '''if "cupy" in sys.modules:
        outcome_cuda_d = np.allclose(gt_double, cuda_double_output)
        outcome_cuda_f = np.allclose(gt_float, cuda_float_output)
        outcome_cuda_grad_d = np.allclose(gt_double_grad, cuda_double_grad)
        outcome_cuda_grad_f = np.allclose(gt_float_grad, cuda_float_grad)
        print("**********\nDid the cuda extension provide the correct result for RBF of "
            f"{xdim}, {num_freqs}? {outcome_cuda_d}")
        print("**********\nDid the cuda extension provide the correct result for RBF of "
            f"{xdim}, {num_freqs}? {outcome_cuda_f}")
        print("**********\nDid the Grad Calc cuda extension provide the correct result for the "
            f"gradient for RBF of {xdim}, {num_freqs}? {outcome_cuda_grad_d}")
        print("**********\nDid the Grad Calc cuda extension provide the correct result for the "
            f"gradient for RBF of {xdim}, {num_freqs}? {outcome_cuda_grad_f}")
        return outcome_d, outcome_f, outcome_cuda_d, outcome_cuda_f, \
                outcome_grad_d, outcome_grad_f, outcome_cuda_grad_d, \
                outcome_cuda_grad_f'''
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

        pretrans_x.append(temp_arr)

    pretrans_x = np.hstack(pretrans_x)[:,:chi_arr.shape[0]]
    pretrans_x *= chi_arr[None,:]

    xtrans = np.zeros((test_array.shape[0], chi_arr.shape[0] * 2))
    gradient = np.zeros((test_array.shape[0], chi_arr.shape[0] * 2, 1))

    for j in range(0, chi_arr.shape[0], 1):
        xtrans[:,2*j] = np.cos(pretrans_x[:,j])
        xtrans[:,2*j+1] = np.sin(pretrans_x[:,j])

        gradient[:,2*j,0] = -xtrans[:,2*j+1] * \
                pretrans_x[:,j]
        gradient[:,2*j+1,0] = xtrans[:,2*j] * \
                pretrans_x[:,j]

    if fit_intercept:
        xtrans *= np.sqrt(1 / (chi_arr.shape[0]-0.5))
        gradient *= np.sqrt(1 / (chi_arr.shape[0]-0.5))
        xtrans[:,0] = 1
        gradient[:,0] = 0
    else:
        xtrans *= np.sqrt(1 / chi_arr.shape[0])
        gradient *= np.sqrt(1 / chi_arr.shape[0])
    return xtrans, gradient


if __name__ == "__main__":
    unittest.main()
