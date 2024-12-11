"""Checks the fast Hadamard transform based routine
used by static layers."""
import sys

import unittest
import numpy as np

from xGPR.xgpr_cpu_rfgen_cpp_ext import cpuConv1dMaxpool
from xGPR.xgpr_cuda_rfgen_cpp_ext import cudaConv1dMaxpool
import cupy as cp

from conv_testing_functions import get_features_maxpool, get_initial_matrices_fht



class TestMaxpoolFeatureGen(unittest.TestCase):
    """Tests feature generation for static layers."""


    def test_conv1d_maxpool(self):
        """Tests the C / Cuda FHT-based convolution with global max pooling
        functions."""
        kernel_width, num_aas, aa_dim, num_freqs = 9, 23, 21, 130
        sigma, ndatapoints = 1, 120
        outcomes = run_maxpool_evaluation(ndatapoints, kernel_width, aa_dim, num_aas,
                    num_freqs, sigma, mode = "maxpool")
        for outcome in outcomes:
            self.assertTrue(outcome)

        kernel_width, num_aas, aa_dim, num_freqs = 15, 23, 1060, 8194
        sigma, ndatapoints = 1, 5
        outcomes = run_maxpool_evaluation(ndatapoints, kernel_width, aa_dim, num_aas,
                    num_freqs, sigma, mode = "maxpool")
        for outcome in outcomes:
            self.assertTrue(outcome)

        kernel_width, num_aas, aa_dim, num_freqs = 5, 56, 2, 62
        sigma, ndatapoints = 1, 100

        outcomes = run_maxpool_evaluation(ndatapoints, kernel_width, aa_dim, num_aas,
                    num_freqs, sigma, mode = "maxpool", precision = "float")
        for outcome in outcomes:
            self.assertTrue(outcome)


        kernel_width, num_aas, aa_dim, num_freqs = 5, 56, 256, 500
        sigma, ndatapoints = 1, 232

        outcomes = run_maxpool_evaluation(ndatapoints, kernel_width, aa_dim, num_aas,
                    num_freqs, sigma, mode = "maxpool")
        for outcome in outcomes:
            self.assertTrue(outcome)

        kernel_width, num_aas, aa_dim, num_freqs = 5, 512, 1024, 1024
        sigma, ndatapoints = 1, 11

        outcomes = run_maxpool_evaluation(ndatapoints, kernel_width, aa_dim, num_aas,
                    num_freqs, sigma, mode = "maxpool")
        for outcome in outcomes:
            self.assertTrue(outcome)


def run_maxpool_evaluation(ndatapoints, kernel_width, aa_dim, num_aas,
                    num_freqs, sigma, mode, precision = "double"):
    """Run an evaluation for the ReLU / maxpool feature generation
    routine, primarily used for static layers."""
    dim2, num_blocks, xdata, seqlen, features, s_mat, \
                radem = get_initial_matrices_fht(ndatapoints, kernel_width,
                        aa_dim, num_aas, num_freqs, mode, precision)
    true_features = get_features_maxpool(xdata, kernel_width, dim2,
                            radem, s_mat, num_freqs, num_blocks,
                            seqlen, precision)

    features = features.astype(np.float32)
    cpuConv1dMaxpool(xdata, features, radem, s_mat, seqlen,
                kernel_width, 2)

    outcome = check_results(true_features, features, precision)
    print(f"Settings: N {ndatapoints}, kernel_width {kernel_width}, "
        f"aa_dim: {aa_dim}, num_aas: {num_aas}, num_freqs: {num_freqs}, "
        f"sigma: {sigma}, mode: {mode}, precision {precision}\n"
        f"Does result match on CPU? {outcome}")

    if "cupy" not in sys.modules:
        return [outcome]

    xdata = cp.asarray(xdata)
    features[:] = 0
    features = cp.asarray(features)
    s_mat = cp.asarray(s_mat)
    radem = cp.asarray(radem)

    cudaConv1dMaxpool(xdata, features, radem, s_mat, seqlen,
                kernel_width)
    outcome_cuda = check_results(true_features, features, precision)
    print(f"Settings: N {ndatapoints}, kernel_width {kernel_width}, "
        f"aa_dim: {aa_dim}, num_aas: {num_aas}, num_freqs: {num_freqs}, "
        f"sigma: {sigma}, mode: {mode}, precision {precision}\n"
        f"Does result match on cuda? {outcome_cuda}")

    print("\n")
    return outcome, outcome_cuda


def check_results(gt_array, test_array, precision):
    """Checks a ground truth array against a test array. We have
    to use different tolerances for 32-bit vs 64 since 32-bit
    can vary slightly."""
    if precision == "float":
        return np.allclose(gt_array, test_array, rtol=1e-6, atol=1e-6)
    return np.allclose(gt_array, test_array)



if __name__ == "__main__":
    unittest.main()
