"""Checks the fast Hadamard transform based routine
used by static layers."""
import sys

import unittest
import numpy as np

from cpu_rf_gen_module import cpuConv1dFGen, cpuConvGrad, cpuConv1dMaxpool
try:
    from cuda_rf_gen_module import gpuConv1dFGen, gpuConvGrad, gpuConv1dMaxpool
    import cupy as cp
except:
    pass

from conv_testing_functions import get_features_maxpool, get_initial_matrices_fht



class TestMaxpoolFeatureGen(unittest.TestCase):
    """Tests feature generation for static layers."""


    def test_conv1d_maxpool(self):
        """Tests the C / Cuda FHT-based convolution with global max pooling
        functions."""
        kernel_width, num_aas, aa_dim, num_freqs = 9, 23, 21, 128
        sigma, ndatapoints = 1, 124
        outcomes = run_maxpool_evaluation(ndatapoints, kernel_width, aa_dim, num_aas,
                    num_freqs, sigma, mode = "maxpool")
        for outcome in outcomes:
            self.assertTrue(outcome)

        kernel_width, num_aas, aa_dim, num_freqs = 5, 56, 2, 62
        sigma, ndatapoints = 1, 1000

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

    cpuConv1dMaxpool(xdata, seqlen, radem, features,
                s_mat, kernel_width, 2)

    outcome = check_results(true_features, features[:,:num_freqs], precision)
    print(f"Settings: N {ndatapoints}, kernel_width {kernel_width}, "
        f"aa_dim: {aa_dim}, num_aas: {num_aas}, num_freqs: {num_freqs}, "
        f"sigma: {sigma}, mode: {mode}, precision {precision}\n"
        f"Does result match on CPU? {outcome}")

    if "cupy" not in sys.modules:
        return [outcome]

    xdata, seqlen = cp.asarray(xdata), cp.asarray(seqlen)
    features[:] = 0
    features = cp.asarray(features)
    s_mat = cp.asarray(s_mat)
    radem = cp.asarray(radem)

    gpuConv1dMaxpool(xdata, seqlen, radem, features,
                s_mat, kernel_width, 2)


    outcome_cuda = check_results(true_features, features[:,:num_freqs], precision)
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
