"""Tests the Cuda and C++ code for computing class-specific means
and prepping input features for pooled covariance calculations."""
import sys
import unittest
import numpy as np
from scipy.linalg import hadamard
import cupy as cp

from xGPR.xgpr_cpu_rfgen_cpp_ext import cpuFindClassMeans, cpuPrepPooledCovCalc
from xGPR.xgpr_cuda_rfgen_cpp_ext import cudaFindClassMeans, cudaPrepPooledCovCalc


class TestClassificationPrep(unittest.TestCase):
    """Run tests on the basic C++ and cuda code for prepping for
    pooled covariance matrix calculations."""


    def test_class_means(self):
        """Tests iterative class means calculations
        on both cpu and cuda."""
        rng = np.random.default_rng(123)
        for nclasses in [1,3,10,30]:
            for arr_dim1 in [2,25,58,1025]:
                true_cl_sums, true_cl_counts = np.zeros((nclasses, arr_dim1)), \
                        np.zeros((nclasses), dtype=np.int64)
                cpu_cl_sums, cpu_cl_counts = \
                        np.zeros((nclasses, arr_dim1)), \
                        np.zeros((nclasses), dtype=np.int64)
                cuda_cl_sums, cuda_cl_counts = \
                        cp.zeros((nclasses, arr_dim1)), \
                        cp.zeros((nclasses), dtype=cp.int64)

                for arr_dim0 in [25,46,360,1024]:
                    rand_array = rng.normal(size=(arr_dim0, arr_dim1))
                    class_array = rng.choice(nclasses, size=arr_dim0).astype(np.int32)

                    for i in range(nclasses):
                        class_idx = class_array == i
                        true_cl_sums[i,:] += rand_array[class_idx,:].sum(axis=0)
                        true_cl_counts[i] += class_idx.sum()

                    cpuFindClassMeans(rand_array, cpu_cl_sums,
                            class_array, cpu_cl_counts)
                    rand_array, class_array = cp.asarray(rand_array), \
                            cp.asarray(class_array)
                    cudaFindClassMeans(rand_array, cuda_cl_sums,
                            class_array, cuda_cl_counts)

                cuda_cl_sums = cp.asnumpy(cuda_cl_sums)
                cuda_cl_counts = cp.asnumpy(cuda_cl_counts)
                self.assertTrue(np.allclose(cuda_cl_sums, true_cl_sums))
                self.assertTrue(np.allclose(cuda_cl_counts, true_cl_counts))
                self.assertTrue(np.allclose(cpu_cl_sums, true_cl_sums))
                self.assertTrue(np.allclose(cpu_cl_counts, true_cl_counts))


    def test_class_cov_prep(self):
        """Tests iterative covariance matrix preparation
        for cpu and cuda."""
        rng = np.random.default_rng(123)
        for nclasses in [1,3,10,30]:
            for arr_dim1 in [2,25,58,1025]:
                for arr_dim0 in [25,46,360,125]:
                    rand_array = rng.normal(size=(arr_dim0, arr_dim1))
                    cp_rand_array = cp.asarray(rand_array)
                    class_array = rng.choice(nclasses, size=arr_dim0).astype(np.int32)
                    class_means = rng.normal(size=(nclasses, arr_dim1))
                    priors = rng.uniform(size=nclasses)

                    true_cov = np.zeros((arr_dim1, arr_dim1))
                    for j in range(arr_dim0):
                        x_mod = rand_array[j,:] - class_means[class_array[j],:]
                        true_cov += np.outer(x_mod, x_mod) * priors[class_array[j]]

                    cpuPrepPooledCovCalc(rand_array, class_means,
                            class_array, np.sqrt(priors))
                    class_means = cp.asarray(class_means)
                    class_array = cp.asarray(class_array)
                    priors = cp.sqrt(cp.asarray(priors))
                    cudaPrepPooledCovCalc(cp_rand_array, class_means,
                            class_array, priors)

                    cpu_cov = rand_array.T @ rand_array
                    cuda_cov = cp_rand_array.T @ cp_rand_array
                    self.assertTrue(np.allclose(cpu_cov, true_cov))
                    self.assertTrue(np.allclose(cuda_cov, true_cov))


if __name__ == "__main__":
    unittest.main()
