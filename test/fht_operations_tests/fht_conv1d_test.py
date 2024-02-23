"""Checks the fast Hadamard transform based convolution
feature generation routines to ensure they are producing
correct results by comparing with a clunky, simple
mostly Python implementation."""
import sys

import unittest
import numpy as np

from cpu_rf_gen_module import cpuConv1dFGen, cpuConvGrad, cpuConv1dMaxpool
try:
    from cuda_rf_gen_module import gpuConv1dFGen, gpuConvGrad, gpuConv1dMaxpool
    import cupy as cp
except:
    pass

from conv_testing_functions import get_initial_matrices_fht, get_features
from conv_testing_functions import get_features_with_gradient



class TestConv1d(unittest.TestCase):
    """Tests FHT-based convolution feature generation."""

    def test_conv1d(self):
        """Tests the FHT-based Conv1d C / Cuda functions."""
        kernel_width, num_aas, aa_dim, num_freqs = 9, 23, 21, 1000
        sigma, ndatapoints = 1, 124

        outcomes = run_basic_eval(ndatapoints, kernel_width, aa_dim, num_aas,
                    num_freqs, sigma)
        for outcome in outcomes:
            self.assertTrue(outcome)
        
        outcomes = run_basic_eval(ndatapoints, kernel_width, aa_dim, num_aas,
                    num_freqs, sigma, precision = "float")
        for outcome in outcomes:
            self.assertTrue(outcome)

        kernel_width, num_aas, aa_dim, num_freqs = 5, 56, 2, 62
        sigma, ndatapoints = 1, 2000

        outcomes = run_basic_eval(ndatapoints, kernel_width, aa_dim, num_aas,
                    num_freqs, sigma)
        for outcome in outcomes:
            self.assertTrue(outcome)

        outcomes = run_basic_eval(ndatapoints, kernel_width, aa_dim, num_aas,
                    num_freqs, sigma, precision = "float")
        for outcome in outcomes:
            self.assertTrue(outcome)

        kernel_width, num_aas, aa_dim, num_freqs = 7, 202, 105, 784
        sigma, ndatapoints = 1, 38

        outcomes = run_basic_eval(ndatapoints, kernel_width, aa_dim, num_aas,
                    num_freqs, sigma)
        for outcome in outcomes:
            self.assertTrue(outcome)

        outcomes = run_basic_eval(ndatapoints, kernel_width, aa_dim, num_aas,
                    num_freqs, sigma, precision = "float")
        for outcome in outcomes:
            self.assertTrue(outcome)

        kernel_width, num_aas, aa_dim, num_freqs = 1, 56, 2, 151
        sigma, ndatapoints = 1, 1000

        outcomes = run_basic_eval(ndatapoints, kernel_width, aa_dim, num_aas,
                    num_freqs, sigma)
        for outcome in outcomes:
            self.assertTrue(outcome)

        kernel_width, num_aas, aa_dim, num_freqs = 10, 11, 200, 784
        sigma, ndatapoints = 1, 38

        outcomes = run_basic_eval(ndatapoints, kernel_width, aa_dim, num_aas,
                    num_freqs, sigma)
        for outcome in outcomes:
            self.assertTrue(outcome)

        outcomes = run_basic_eval(ndatapoints, kernel_width, aa_dim, num_aas,
                    num_freqs, sigma, precision = "float")
        for outcome in outcomes:
            self.assertTrue(outcome)



    def test_conv1d_gradients(self):
        """Tests the gradient calculations for the FHT-based Cuda / C functions."""
        kernel_width, num_aas, aa_dim, num_freqs = 9, 23, 21, 128
        sigma, ndatapoints = 1, 36

        outcomes = run_gradient_eval(ndatapoints, kernel_width, aa_dim, num_aas,
                    num_freqs, sigma)
        for outcome in outcomes:
            self.assertTrue(outcome)

        outcomes = run_gradient_eval(ndatapoints, kernel_width, aa_dim, num_aas,
                    num_freqs, sigma, precision = "float")
        for outcome in outcomes:
            self.assertTrue(outcome)


        kernel_width, num_aas, aa_dim, num_freqs = 5, 56, 2, 62
        sigma, ndatapoints = 1, 53

        outcomes = run_gradient_eval(ndatapoints, kernel_width, aa_dim, num_aas,
                    num_freqs, sigma)
        for outcome in outcomes:
            self.assertTrue(outcome)

        outcomes = run_gradient_eval(ndatapoints, kernel_width, aa_dim, num_aas,
                    num_freqs, sigma, precision = "float")
        for outcome in outcomes:
            self.assertTrue(outcome)




def run_basic_eval(ndatapoints, kernel_width, aa_dim, num_aas,
        num_freqs, sigma, precision = "double"):
    """Run an evaluation of RBF-based convolution kernel feature
    evaluation, without evaluating gradient."""
    dim2, num_blocks, xdata, seqlen, features, s_mat, \
                radem = get_initial_matrices_fht(ndatapoints, kernel_width,
                        aa_dim, num_aas, num_freqs, "conv", precision)
    true_features = get_features(xdata, kernel_width, dim2,
                            radem, s_mat, num_freqs, num_blocks, sigma,
                            seqlen, precision)
    xd = xdata * sigma
    cpuConv1dFGen(xd, seqlen, radem, features, s_mat,
            kernel_width, 2)

    outcome = check_results(true_features, features, precision)
    print(f"Settings: N {ndatapoints}, kernel_width {kernel_width}, "
        f"aa_dim: {aa_dim}, num_aas: {num_aas}, num_freqs: {num_freqs}, "
        f"sigma: {sigma}, mode: RBF convolution, precision {precision}\n"
        f"Does result match on CPU? {outcome}")

    if "cupy" not in sys.modules:
        return [outcome]
    xd, seqlen = cp.asarray(xd), cp.asarray(seqlen)
    features[:] = 0
    features = cp.asarray(features)
    s_mat = cp.asarray(s_mat)
    radem = cp.asarray(radem)
    gpuConv1dFGen(xd, seqlen, radem, features, s_mat,
            kernel_width, 2)

    features = cp.asnumpy(features)
    outcome_cuda = check_results(true_features, features, precision)
    print(f"Settings: N {ndatapoints}, kernel_width {kernel_width}, "
        f"aa_dim: {aa_dim}, num_aas: {num_aas}, num_freqs: {num_freqs}, "
        f"sigma: {sigma}, mode: RBF convolution, precision {precision}\n"
        f"Does result match on cuda? {outcome_cuda}")

    print("\n")
    return outcome, outcome_cuda



def run_gradient_eval(ndatapoints, kernel_width, aa_dim, num_aas,
                    num_freqs, sigma, precision = "double"):
    """Evaluate RBF-based convolution kernel feature evaluation AND
    gradient calculation."""
    dim2, num_blocks, xdata, reshaped_x, features, s_mat, \
                radem = get_initial_matrices_fht(ndatapoints, kernel_width,
                        aa_dim, num_aas, num_freqs, "conv_gradient", precision)
    true_features, true_gradient = get_features_with_gradient(xdata,
                            kernel_width, dim2, radem, s_mat, num_freqs,
                            num_blocks, sigma, precision,
                            False)

    seqlen = np.full(xdata.shape[0], xdata.shape[1]).astype(np.int32)
    xd = xdata * sigma
    gradient = cpuConvGrad(xd, seqlen, radem, features, s_mat,
            kernel_width, 2, sigma)
    gradient = gradient[:,:(2*num_freqs),0]


    outcome = check_results(true_features, features[:,:(2 * num_freqs)], precision)
    outcome_gradient = check_results(true_gradient, gradient, precision)
    print(f"Settings: N {ndatapoints}, kernel_width {kernel_width}, "
            f"aa_dim: {aa_dim}, num_aas: {num_aas}, num_freqs: {num_freqs}, "
            f"sigma: {sigma}, mode: RBF conv gradient, precision: {precision}\n"
            f"Does result match on CPU? {outcome}\n"
            f"Does gradient match on CPU? {outcome_gradient}")

    if "cupy" not in sys.modules:
        return outcome, outcome_gradient

    xd, seqlen = cp.asarray(xd), cp.asarray(seqlen)
    features[:] = 0
    features = cp.asarray(features)
    s_mat = cp.asarray(s_mat)
    radem = cp.asarray(radem)

    gradient = gpuConvGrad(xd, seqlen, radem, features, s_mat,
            kernel_width, 2, sigma)
    features = cp.asnumpy(features[:,:(2*num_freqs)])
    gradient = cp.asnumpy(gradient[:,:(2*num_freqs),0])

    outcome_cuda = check_results(true_features, features, precision)
    outcome_cuda_gradient = check_results(true_gradient, gradient, precision)
    print(f"Settings: N {ndatapoints}, kernel_width {kernel_width}, "
        f"aa_dim: {aa_dim}, num_aas: {num_aas}, num_freqs: {num_freqs}, "
        f"sigma: {sigma}, mode: RBF conv gradient, precision {precision}\n"
        f"Does result match on cuda? {outcome_cuda}\n"
        f"Does gradient match on cuda? {outcome_cuda_gradient}\n")

    print("\n")
    return outcome, outcome_gradient, outcome_cuda, outcome_cuda_gradient





def check_results(gt_array, test_array, precision):
    """Checks a ground truth array against a test array. We have
    to use different tolerances for 32-bit vs 64 since 32-bit
    can vary slightly."""
    if precision == "float":
        return np.allclose(gt_array, test_array, rtol=1e-6, atol=1e-6)
    return np.allclose(gt_array, test_array)



if __name__ == "__main__":
    unittest.main()
