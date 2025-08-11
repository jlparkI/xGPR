"""Tests the Cuda and C++ code for computing class-specific means
and prepping input features for pooled covariance calculations."""
import sys
import os
import unittest
import numpy as np
import cupy as cp

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from xGPR import build_classification_dataset
from xGPR.xgpr_cpu_rfgen_cpp_ext import cpuFindClassMeans, cpuPrepPooledCovCalc, cpu_mean_variance
from xGPR.xgpr_cuda_rfgen_cpp_ext import cudaFindClassMeans, cudaPrepPooledCovCalc
from utils.build_test_dataset import build_test_dataset
from utils.model_constructor import get_discriminant_models



class TestClassificationPrep(unittest.TestCase):
    """Run tests on the basic C++ and cuda code for prepping for
    pooled covariance matrix calculations."""


    def test_class_means(self):
        """Tests iterative class means calculations
        on both cpu and cuda."""
        return
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
        return
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
    

    def test_welford_mean_var(self):
        """Tests the recursive mean and variance calculation
        routine for CPU."""
        # First, compare both online and offline using a small test
        # dataset. First compare using a linear kernel, under which
        # the input values are not transformed, and then using
        # an RBF kernel, under which they are.
        online_data, offline_data = build_test_dataset()
        full_x = np.vstack([x for (x,_) in
            online_data.get_chunked_x_data()])

        gt_xmean = np.zeros((full_x.shape[1]+1))
        gt_xvar = gt_xmean.copy()
        gt_xmean[1:] = full_x.mean(axis=0)
        gt_xvar[1:] = full_x.var(axis=0, ddof=1).clip(min=1e-14)
        gt_xvar[0] = 1

        cpu_mod, _ = get_discriminant_models("Linear", online_data,
                num_rffs = 1024)

        cpu_mod._set_data_mean_var(offline_data)
        xmean, xvar = cpu_mod.kernel.get_xmean_xvar()
        self.assertTrue(np.allclose(xmean, gt_xmean))
        self.assertTrue(np.allclose(xvar, gt_xvar))
        cpu_mod.kernel.reset_xmean_xvar()

        cpu_mod._set_data_mean_var(online_data)
        xmean, xvar = cpu_mod.kernel.get_xmean_xvar()
        self.assertTrue(np.allclose(xmean, gt_xmean))
        self.assertTrue(np.allclose(xvar, gt_xvar))
        cpu_mod.kernel.reset_xmean_xvar()

        # Now try the RBF kernel.
        cpu_mod, _ = get_discriminant_models("RBF", online_data,
                num_rffs = 1026)
        gt_xmean = np.zeros((cpu_mod.kernel.get_num_rffs()))
        gt_xvar = gt_xmean.copy()
        full_x = np.vstack([cpu_mod.kernel.transform_x(x[0]) for
            x in online_data.get_chunked_x_data()])
        gt_xmean = full_x.mean(axis=0)
        gt_xvar = full_x.var(axis=0, ddof=1).clip(min=1e-14)
        gt_xvar[0] = 1
        gt_xmean[0] = 0

        cpu_mod._set_data_mean_var(offline_data)
        xmean, xvar = cpu_mod.kernel.get_xmean_xvar()
        self.assertTrue(np.allclose(xmean, gt_xmean))
        self.assertTrue(np.allclose(xvar, gt_xvar))
        cpu_mod.kernel.reset_xmean_xvar()

        cpu_mod._set_data_mean_var(online_data)
        xmean, xvar = cpu_mod.kernel.get_xmean_xvar()
        self.assertTrue(np.allclose(xmean, gt_xmean))
        self.assertTrue(np.allclose(xvar, gt_xvar))
        cpu_mod.kernel.reset_xmean_xvar()

        # Next, create a larger dataset using random values to ensure
        # that mean / variance calculation is still correct when data
        # is served up in chunks.
        rng = np.random.default_rng(123)
        full_x = rng.uniform(size=(10000,123))
        yvalues = np.ones(10000, dtype=np.int32)
        yvalues[0] = 0
        online_data = build_classification_dataset(full_x, yvalues)

        cpu_mod, _ = get_discriminant_models("RBF", online_data,
                num_rffs = 536)
        gt_xmean = np.zeros((cpu_mod.kernel.get_num_rffs()))
        gt_xvar = gt_xmean.copy()
        full_x = np.vstack([cpu_mod.kernel.transform_x(x[0]) for
            x in online_data.get_chunked_x_data()])
        gt_xmean = full_x.mean(axis=0)
        gt_xvar = full_x.var(axis=0, ddof=1).clip(min=1e-14)
        gt_xvar[0] = 1
        gt_xmean[0] = 0

        cpu_mod._set_data_mean_var(online_data)
        xmean, xvar = cpu_mod.kernel.get_xmean_xvar()
        self.assertTrue(np.allclose(xmean, gt_xmean))
        self.assertTrue(np.allclose(xvar, gt_xvar))
        cpu_mod.kernel.reset_xmean_xvar()




if __name__ == "__main__":
    unittest.main()
