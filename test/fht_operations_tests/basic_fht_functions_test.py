"""Tests the 'basic' hadamard transform based operations (SORF, SRHT, FHT)
for CPU and -- if GPU is available -- for GPU as well. We must test
for both single and double precision."""
import sys
import unittest
import numpy as np
from scipy.linalg import hadamard
from cpu_rf_gen_module import doubleCpuFastHadamardTransform as dFHT
from cpu_rf_gen_module import floatCpuFastHadamardTransform as fFHT

from cpu_rf_gen_module import doubleCpuFastHadamardTransform2D as dFHT2D
from cpu_rf_gen_module import floatCpuFastHadamardTransform2D as fFHT2D

from cpu_rf_gen_module import doubleCpuSORFTransform as dSORF
from cpu_rf_gen_module import floatCpuSORFTransform as fSORF

from cpu_rf_gen_module import doubleCpuSRHT as dSRHT
from cpu_rf_gen_module import floatCpuSRHT as fSRHT

try:
    from cuda_rf_gen_module import cudaPySORFTransform as cudaSORF
    from cuda_rf_gen_module import cudaSRHT
    import cupy as cp
except:
    pass

#To test the C extension for the fast hadamard transform, we compare the
#results to matrix multiplication with Hadamard matrices generated by
#Scipy (result should be the same).
class TestFastHadamardTransform(unittest.TestCase):
    """Runs tests for basic functionality (i.e. FHT and SORF)
    for float and double precision for CPU and (if available) GPU."""


    def test_3d_array_transform(self):
        """Tests the 3d FHT transform. This is for CPU only;
        the cuda module does not provide a separate function
        for FHT only at this time."""
        dim = (124,36,4)
        outcome_f, outcome_d = run_fht_test(dim)
        self.assertTrue(outcome_d)
        self.assertTrue(outcome_f)

        dim = (3001, 5, 1024)
        outcome_f, outcome_d = run_fht_test(dim)
        self.assertTrue(outcome_d)
        self.assertTrue(outcome_f)

        dim = (250, 1, 4096)
        outcome_f, outcome_d = run_fht_test(dim)
        self.assertTrue(outcome_d)
        self.assertTrue(outcome_f)


    def test_2d_array_transform(self):
        """Tests the 2d FHT transform. This is for CPU only;
        the cuda module does not provide a separate function
        for FHT only at this time."""
        dim = (124, 4)
        outcome_f, outcome_d = run_fht_2d_test(dim)
        self.assertTrue(outcome_d)
        self.assertTrue(outcome_f)

        dim = (3001, 1024)
        outcome_f, outcome_d = run_fht_2d_test(dim)
        self.assertTrue(outcome_d)
        self.assertTrue(outcome_f)

        dim = (250, 4096)
        outcome_f, outcome_d = run_fht_2d_test(dim)
        self.assertTrue(outcome_d)
        self.assertTrue(outcome_f)


    def test_block_transform(self):
        """Tests SORF functionality. Note that this tests SORF
        functionality by using FHT. Therefore if the FHT did not
        pass, this one will not either."""

        outcomes = run_sorf_test(1, 2048)
        for outcome in outcomes:
            self.assertTrue(outcome)

        outcomes = run_sorf_test(3, 256)
        for outcome in outcomes:
            self.assertTrue(outcome)

        outcomes = run_sorf_test(30, 128)
        for outcome in outcomes:
            self.assertTrue(outcome)


    def test_srht(self):
        """Tests SRHT functionality. Note that this tests SRHT
        functionality by using FHT. Therefore if the FHT did not
        pass, this one will not either."""

        outcomes = run_srht_test((150,256), 128)
        for outcome in outcomes:
            self.assertTrue(outcome)

        outcomes = run_srht_test((304,512), 256)
        for outcome in outcomes:
            self.assertTrue(outcome)

        outcomes = run_srht_test((5,2048), 512)
        for outcome in outcomes:
            self.assertTrue(outcome)


def run_fht_test(dim, random_seed = 123):
    """A helper function that runs an FHT test with specified
    dimensionality."""
    scipy_double, scipy_float, marr, marr_float = setup_fht_test(dim, random_seed)
    dFHT(marr, 1)
    fFHT(marr_float, 1)
    outcome_d = np.allclose(scipy_double, marr)
    outcome_f = np.allclose(scipy_float, marr_float, rtol=1e-4, atol=1e-4)
    print("**********\nDid the C extension provide the correct result for columnwise "
            f"transforms of a {dim} 3d array of doubles? {outcome_d}\n*******")
    print("**********\nDid the C extension provide the correct result for columnwise "
            f"transforms of a {dim} 3d array of floats? {outcome_f}\n*******")
    return outcome_d, outcome_f



def run_fht_2d_test(dim, random_seed = 123):
    """A helper function that runs an FHT 2d test with specified
    dimensionality."""
    scipy_double, scipy_float, marr, marr_float = setup_fht_2d_test(dim, random_seed)
    dFHT2D(marr, 1)
    fFHT2D(marr_float, 1)
    outcome_d = np.allclose(scipy_double, marr)
    outcome_f = np.allclose(scipy_float, marr_float, rtol=1e-4, atol=1e-4)
    print("**********\nDid the C extension provide the correct result for columnwise "
            f"transforms of a {dim} 2d array of doubles? {outcome_d}\n*******")
    print("**********\nDid the C extension provide the correct result for columnwise "
            f"transforms of a {dim} 2d array of floats? {outcome_f}\n*******")
    return outcome_d, outcome_f


def setup_fht_test(dim, random_seed = 123):
    """A helper function that builds a matrix to test on
    with specified dimensions and gets the 'true' result using
    Scipy's Hadamard matrix generator. This is a very slow
    way to do a transform -- we use it here only to cross-
    check the results from our routines. Generated for
    both single and double precision."""
    rng = np.random.default_rng(random_seed)
    marr = rng.uniform(low=-10.0,high=10.0, size=dim)

    hmat = hadamard(marr.shape[2])
    scipy_results = []
    for i in range(marr.shape[1]):
        scipy_transform = np.matmul(hmat, marr[:,i,:].T).T
        scipy_results.append(scipy_transform)
    scipy_double = np.stack(scipy_results, axis=1)

    marr_float = marr.copy().astype(np.float32)
    scipy_results = []
    for i in range(marr_float.shape[1]):
        scipy_transform = np.matmul(hmat, marr_float[:,i,:].T).T
        scipy_results.append(scipy_transform)
    scipy_float = np.stack(scipy_results, axis=1)
    return scipy_double, scipy_float, marr, marr_float

def setup_fht_2d_test(dim, random_seed = 123):
    """A helper function that builds a matrix to test on
    with specified dimensions and gets the 'true' result using
    Scipy's Hadamard matrix generator. This is a very slow
    way to do a transform -- we use it here only to cross-
    check the results from our routines. Generated for
    both single and double precision."""
    rng = np.random.default_rng(random_seed)
    marr = rng.uniform(low=-10.0,high=10.0, size=dim)

    hmat = hadamard(marr.shape[1])
    scipy_double = np.matmul(hmat, marr.T).T

    marr_float = marr.copy().astype(np.float32)
    scipy_float = np.matmul(hmat, marr_float.T).T
    return scipy_double, scipy_float, marr, marr_float


def run_sorf_test(nblocks, dim2, random_seed = 123):
    """A helper function that runs the SORF test for
    specified input dimensions."""
    marr_gt_double, marr_test_double, marr_gt_float, marr_test_float,\
            radem, norm_constant = setup_sorf_test(nblocks, dim2, random_seed)

    if "cupy" in sys.modules:
        cuda_test_double = cp.asarray(marr_test_double)
        cuda_test_float = cp.asarray(marr_test_float)

    marr_gt_double = marr_gt_double * radem[0:1,:,:] * norm_constant
    dFHT(marr_gt_double, 2)
    marr_gt_double = marr_gt_double * radem[1:2,:,:] * norm_constant
    dFHT(marr_gt_double, 2)
    marr_gt_double = marr_gt_double * radem[2:3,:,:] * norm_constant
    dFHT(marr_gt_double, 2)

    marr_gt_float = marr_gt_float * radem[0:1,:,:] * norm_constant
    fFHT(marr_gt_float, 2)
    marr_gt_float = marr_gt_float * radem[1:2,:,:] * norm_constant
    fFHT(marr_gt_float, 2)
    marr_gt_float = marr_gt_float * radem[2:3,:,:] * norm_constant
    fFHT(marr_gt_float, 2)

    dSORF(marr_test_double, radem, 2)
    fSORF(marr_test_float, radem, 2)
    outcome_d = np.allclose(marr_gt_double, marr_test_double)
    outcome_f = np.allclose(marr_gt_float, marr_test_float)
    print("**********\nDid the C extension provide the correct result for SORF of "
            f"an {nblocks}, {dim2} 3d array of doubles? {outcome_d}\n*******")
    print("**********\nDid the C extension provide the correct result for SORF of "
            f"an {nblocks}, {dim2} 3d array of floats? {outcome_f}\n*******")

    if "cupy" in sys.modules:
        radem = cp.asarray(radem)
        cudaSORF(cuda_test_double, radem, 2)
        cudaSORF(cuda_test_float, radem, 2)
        cuda_test_double = cp.asnumpy(cuda_test_double)
        cuda_test_float = cp.asnumpy(cuda_test_float)
        outcome_cuda_d = np.allclose(marr_gt_double, cuda_test_double)
        outcome_cuda_f = np.allclose(marr_gt_float, cuda_test_float)
        print("**********\nDid the Cuda extension provide the correct result for SORF of "
            f"an {nblocks}, {dim2} 3d array of doubles? {outcome_cuda_d}\n*******")
        print("**********\nDid the Cuda extension provide the correct result for SORF of "
            f"an {nblocks}, {dim2} 3d array of floats? {outcome_cuda_f}\n*******")
        return outcome_d, outcome_f, outcome_cuda_d, outcome_cuda_f
    return outcome_d, outcome_f



def setup_sorf_test(nblocks, dim2, random_seed = 123):
    """A helper function that builds the matrices required for
    the SORF test, specified using the input dimensions."""
    radem_array = np.asarray([-1.0,1.0], dtype=np.int8)
    rng = np.random.default_rng(random_seed)

    marr_gt_double = rng.uniform(low=-10.0,high=10.0, size=(2000,nblocks,dim2))
    marr_test_double = marr_gt_double.copy()
    marr_gt_float = marr_gt_double.copy().astype(np.float32)
    marr_test_float = marr_gt_float.copy()

    radem = rng.choice(radem_array, size=(3,nblocks,dim2), replace=True)
    norm_constant = np.log2(dim2) / 2
    norm_constant = 1 / (2**norm_constant)
    return marr_gt_double, marr_test_double, marr_gt_float, marr_test_float, \
            radem, norm_constant



def run_srht_test(dim, compression_size, random_seed = 123):
    """A helper function that runs the SRHT test for
    specified input dimensions."""
    marr_gt_double, marr_test_double, marr_gt_float, marr_test_float,\
            radem, norm_constant, sampler, scaling_factor = setup_srht_test(dim,
                                    compression_size, random_seed)

    if "cupy" in sys.modules:
        cuda_test_double = cp.asarray(marr_test_double)
        cuda_test_float = cp.asarray(marr_test_float)

    marr_gt_double = marr_gt_double * radem[None,:] * norm_constant
    dFHT2D(marr_gt_double, 2)
    marr_gt_double[:,:compression_size] = marr_gt_double[:,sampler]
    marr_gt_double[:,:compression_size] *= scaling_factor

    marr_gt_float = marr_gt_float * radem[None,:] * norm_constant
    fFHT2D(marr_gt_float, 2)
    marr_gt_float[:,:compression_size] = marr_gt_float[:,sampler]
    marr_gt_float[:,:compression_size] *= scaling_factor

    dSRHT(marr_test_double, radem, sampler, compression_size, 2)
    fSRHT(marr_test_float, radem, sampler, compression_size, 2)
    outcome_d = np.allclose(marr_gt_double, marr_test_double)
    outcome_f = np.allclose(marr_gt_float, marr_test_float)
    print("**********\nDid the C extension provide the correct result for SRHT of "
            f"a {dim} 2d array of doubles? {outcome_d}\n*******")
    print("**********\nDid the C extension provide the correct result for SRHT of "
            f"a {dim} 2d array of floats? {outcome_f}\n*******")

    if "cupy" in sys.modules:
        radem = cp.asarray(radem)
        cudaSRHT(cuda_test_float, radem, sampler, compression_size, 2)
        cudaSRHT(cuda_test_double, radem, sampler, compression_size, 2)
        cuda_test_double = cp.asnumpy(cuda_test_double)
        cuda_test_float = cp.asnumpy(cuda_test_float)
        outcome_cuda_d = np.allclose(marr_gt_double, cuda_test_double)
        outcome_cuda_f = np.allclose(marr_gt_float, cuda_test_float)
        print("**********\nDid the Cuda extension provide the correct result for SRHT of "
            f"a {dim} 2d array of doubles? {outcome_cuda_d}\n*******")
        print("**********\nDid the Cuda extension provide the correct result for SRHT of "
            f"a {dim} 2d array of floats? {outcome_cuda_f}\n*******")
        return outcome_d, outcome_f, outcome_cuda_d, outcome_cuda_f

    return outcome_d, outcome_f



def setup_srht_test(dim, compression_size, random_seed = 123):
    """A helper function that builds the matrices required for
    the SRHT test, specified using the input dimensions."""
    radem_array = np.asarray([-1.0,1.0], dtype=np.int8)
    rng = np.random.default_rng(random_seed)

    marr_gt_double = rng.uniform(low=-10.0,high=10.0, size=dim)
    marr_test_double = marr_gt_double.copy()
    marr_gt_float = marr_gt_double.copy().astype(np.float32)
    marr_test_float = marr_gt_float.copy()

    radem = rng.choice(radem_array, size=(dim[1]), replace=True)
    sampler = rng.permutation(dim[1])[:compression_size]
    norm_constant = np.log2(dim[1]) / 2
    norm_constant = 1 / (2**norm_constant)
    scaling_factor = np.sqrt(radem.shape[0] / compression_size)
    return marr_gt_double, marr_test_double, marr_gt_float, marr_test_float, \
            radem, norm_constant, sampler, scaling_factor


if __name__ == "__main__":
    unittest.main()
