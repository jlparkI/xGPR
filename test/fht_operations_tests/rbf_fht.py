"""Tests the RBF feature generation routines (specific for RBF, Matern
and MiniARD / ARD kernels, which by extension includes static layer kernels."""
import sys
import unittest
import timeit
from math import ceil
import numpy as np
from scipy.stats import chi
from cpu_basic_hadamard_operations import doubleCpuRBFFeatureGen as dRBF
from cpu_basic_hadamard_operations import floatCpuRBFFeatureGen as fRBF

from cpu_basic_hadamard_operations import doubleCpuSORFTransform as dSORF
from cpu_basic_hadamard_operations import floatCpuSORFTransform as fSORF

try:
    from cuda_basic_hadamard_operations import doubleCudaRBFFeatureGen as dCudaRBF
    from cuda_basic_hadamard_operations import floatCudaRBFFeatureGen as fCudaRBF
    import cupy as cp
except:
    pass


class TestRBFFeatureGen(unittest.TestCase):
    """Runs tests for RBF feature generation for float and double
    precision for CPU and (if available) GPU. We do this by comparing
    with feature generation using a Python routine that
    makes use of SORFTransform functions, so the basic_fht test must
    pass in order for this test to work."""


    def test_rbf_feature_gen(self):
        """Tests RBF feature generation for CPU and if available GPU."""
        outcomes = run_rbf_test((10,50), 2000, beta_value = 1.84)
        for outcome in outcomes:
            self.assertTrue(outcome)

        outcomes = run_rbf_test((1000,232), 1000, beta_value = 0.606)
        for outcome in outcomes:
            self.assertTrue(outcome)

        outcomes = run_rbf_test((512,856), 500, beta_value = 1.105)
        for outcome in outcomes:
            self.assertTrue(outcome)

def time_test():
    """Compare speed of the two variants."""
    ntests = 10
    block_setup = f"""
import numpy as np
random_seed = 123
from __main__ import setup_rbf_test
_, test_float, radem, _, chi_float = setup_rbf_test((1000,512), 2000, 123)
from __main__ import generate_float_rbf_values
"""
    time_taken = timeit.timeit("generate_float_rbf_values(test_float, radem, chi_float, 1.0)", setup=block_setup,
                number=ntests)
    print(f"Time for old variant: {1e6 * time_taken / ntests}")
    block_setup = f"""
import numpy as np
from cpu_basic_hadamard_operations import floatCpuRBFFeatureGen as fRBF
random_seed = 123
from __main__ import setup_rbf_test
_, test_float, radem, _, chi_float = setup_rbf_test((1000,512), 2000, 123)
float_out = np.zeros((1000,4000))
"""
    time_taken = timeit.timeit("fRBF(test_float, float_out, radem, chi_float, 1.0, 2000, 3)", setup=block_setup,
                number=ntests)
    print(f"Time for new variant: {1e6 * time_taken / ntests}")



def run_rbf_test(xdim, num_freqs, random_seed = 123, beta_value = 1):
    """A helper function that runs the RBF test for
    specified input dimensions."""

    test_array, test_float, radem, \
            chi_arr, chi_float = setup_rbf_test(xdim, num_freqs, random_seed)

    gt_double = generate_double_rbf_values(test_array, radem, chi_arr, beta_value)
    gt_float = generate_float_rbf_values(test_float, radem, chi_float, beta_value)

    temp_test = test_array.copy()
    double_output = np.zeros((test_array.shape[0], num_freqs * 2))
    dRBF(temp_test, double_output, radem, chi_arr, beta_value, num_freqs, 2)

    temp_test = test_float.copy()
    float_output = np.zeros((test_array.shape[0], num_freqs * 2))
    fRBF(temp_test, float_output, radem, chi_float, beta_value, num_freqs, 2)

    if "cupy" in sys.modules:
        cuda_test_array = cp.asarray(test_array)
        cuda_test_float = cp.asarray(test_float)
        radem = cp.asarray(radem)
        chi_arr = cp.asarray(chi_arr)
        chi_float = cp.asarray(chi_float)
        cuda_double_output = cp.zeros((test_array.shape[0], num_freqs * 2))
        cuda_float_output = cp.zeros((test_array.shape[0], num_freqs * 2))

        dCudaRBF(cuda_test_array, cuda_double_output, radem,
                chi_arr, beta_value, num_freqs, 2)
        fCudaRBF(cuda_test_float, cuda_float_output, radem,
                chi_float, beta_value, num_freqs, 2)


    outcome_d = np.allclose(gt_double, double_output)
    outcome_f = np.allclose(gt_float, float_output)
    print("**********\nDid the C extension provide the correct result for RBF of "
            f"{xdim}, {num_freqs}? {outcome_d}\n*******")
    print("**********\nDid the C extension provide the correct result for RBF of "
            f"{xdim}, {num_freqs}? {outcome_f}\n*******")

    if "cupy" in sys.modules:
        outcome_cuda_d = np.allclose(gt_double, cuda_double_output)
        outcome_cuda_f = np.allclose(gt_float, cuda_float_output)
        print("**********\nDid the cuda extension provide the correct result for RBF of "
            f"{xdim}, {num_freqs}? {outcome_cuda_d}\n*******")
        print("**********\nDid the cuda extension provide the correct result for RBF of "
            f"{xdim}, {num_freqs}? {outcome_cuda_f}\n*******")
        return outcome_d, outcome_f, outcome_cuda_d, outcome_cuda_f
    return outcome_d, outcome_f



def setup_rbf_test(xdim, num_freqs, random_seed = 123):
    """A helper function that builds the matrices required for
    the RBF test, specified using the input dimensions."""
    padded_dims = 2**ceil(np.log2(max(xdim[-1], 2)))
    if padded_dims < num_freqs:
        nblocks = ceil(num_freqs / padded_dims)
    else:
        nblocks = 1

    radem_array = np.asarray([-1.0,1.0], dtype=np.int8)
    rng = np.random.default_rng(random_seed)


    radem = rng.choice(radem_array, size=(3,nblocks,padded_dims), replace=True)
    chi_arr = chi.rvs(df=padded_dims, size=num_freqs,
                            random_state = random_seed)
    chi_float = chi_arr.astype(np.float32)

    init_array = rng.uniform(low=-10.0,high=10.0, size=(xdim[0], xdim[1]))
    test_array = np.zeros((init_array.shape[0], radem.shape[1], radem.shape[2]))
    test_array[:,:,:init_array.shape[1]] = init_array[:,None,:]
    test_float = test_array.astype(np.float32)
    return test_array, test_float, radem, chi_arr, chi_float


def generate_double_rbf_values(test_array, radem, chi_arr, beta):
    """Generates the 'ground-truth' RBF values for
    comparison with those from the C / Cuda extension."""
    pretrans_x = test_array.copy()
    dSORF(pretrans_x, radem, 2)
    pretrans_x = pretrans_x.reshape((pretrans_x.shape[0], pretrans_x.shape[1] *
                        pretrans_x.shape[2]))[:,:chi_arr.shape[0]]
    pretrans_x *= chi_arr[None,:]

    xtrans = np.zeros((test_array.shape[0], chi_arr.shape[0] * 2))
    xtrans[:,:chi_arr.shape[0]] = np.cos(pretrans_x)
    xtrans[:,chi_arr.shape[0]:] = np.sin(pretrans_x)
    xtrans *= (beta * np.sqrt(1 / chi_arr.shape[0]))
    return xtrans


def generate_float_rbf_values(test_array, radem, chi_arr, beta):
    """Generates the 'ground-truth' RBF values for
    comparison with those from the C / Cuda extension."""
    pretrans_x = test_array.copy()
    fSORF(pretrans_x, radem, 3)
    pretrans_x = pretrans_x.reshape((pretrans_x.shape[0], pretrans_x.shape[1] *
                        pretrans_x.shape[2]))[:,:chi_arr.shape[0]]
    pretrans_x *= chi_arr[None,:]

    xtrans = np.zeros((test_array.shape[0], chi_arr.shape[0] * 2))
    xtrans[:,:chi_arr.shape[0]] = np.cos(pretrans_x)
    xtrans[:,chi_arr.shape[0]:] = np.sin(pretrans_x)
    xtrans *= (beta * np.sqrt(1 / chi_arr.shape[0]))
    return xtrans


if __name__ == "__main__":
    #time_test()
    unittest.main()
