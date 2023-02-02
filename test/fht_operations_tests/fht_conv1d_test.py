"""Checks the fast Hadamard transform based convolution
feature generation routines to ensure they are producing
correct results by comparing with a clunky, simple
mostly Python implementation."""
import sys

import unittest
import numpy as np

from cpu_convolution_double_hadamard_operations import doubleCpuConv1dTransform
from cpu_convolution_float_hadamard_operations import floatCpuConv1dTransform
try:
    from cuda_convolution_double_hadamard_operations import doubleGpuConv1dTransform
    from cuda_convolution_float_hadamard_operations import floatGpuConv1dTransform
    import cupy as cp
except:
    pass

from conv_testing_functions import get_initial_matrices_fht, get_features
from conv_testing_functions import get_features_maxpool, get_features_with_gradient



class TestConv1d(unittest.TestCase):
    """Tests FHT-based convolution feature generation."""

    def test_conv1d(self):
        """Tests the FHT-based Conv1d C / Cuda functions."""
        kernel_width = 9
        num_aas = 23
        aa_dim = 21
        num_freqs = 1000
        sigma = 1
        ndatapoints = 124

        outcomes = run_evaluation(ndatapoints, kernel_width, aa_dim, num_aas,
                    num_freqs, sigma, mode = "conv")
        for outcome in outcomes:
            self.assertTrue(outcome)

        kernel_width = 5
        num_aas = 56
        aa_dim = 2
        num_freqs = 62
        sigma = 1
        ndatapoints = 2000

        outcomes = run_evaluation(ndatapoints, kernel_width, aa_dim, num_aas,
                    num_freqs, sigma, mode = "conv")
        for outcome in outcomes:
            self.assertTrue(outcome)

        outcomes = run_evaluation(ndatapoints, kernel_width, aa_dim, num_aas,
                    num_freqs, sigma, mode = "conv", precision = "float")
        for outcome in outcomes:
            self.assertTrue(outcome)



    def test_conv1d_gradients(self):
        """Tests the gradient calculations for the FHT-based Cuda / C functions."""
        kernel_width = 9
        num_aas = 23
        aa_dim = 21
        num_freqs = 128
        sigma = 1
        ndatapoints = 36

        outcomes = run_evaluation(ndatapoints, kernel_width, aa_dim, num_aas,
                    num_freqs, sigma, mode = "conv_gradient")
        for outcome in outcomes:
            self.assertTrue(outcome)

        kernel_width = 5
        num_aas = 56
        aa_dim = 2
        num_freqs = 62
        sigma = 1
        ndatapoints = 1


        outcomes = run_evaluation(ndatapoints, kernel_width, aa_dim, num_aas,
                    num_freqs, sigma, mode = "conv_gradient")
        for outcome in outcomes:
            self.assertTrue(outcome)

        outcomes = run_evaluation(ndatapoints, kernel_width, aa_dim, num_aas,
                    num_freqs, sigma, mode = "conv_gradient", precision = "float")
        for outcome in outcomes:
            self.assertTrue(outcome)


    def test_conv1d_maxpool(self):
        """Tests the C / Cuda FHT-based convolution with global max pooling
        functions."""
        kernel_width = 9
        num_aas = 23
        aa_dim = 21
        num_freqs = 128
        sigma = 1
        ndatapoints = 124
        outcomes = run_evaluation(ndatapoints, kernel_width, aa_dim, num_aas,
                    num_freqs, sigma, mode = "maxpool")
        for outcome in outcomes:
            self.assertTrue(outcome)

        kernel_width = 5
        num_aas = 56
        aa_dim = 2
        num_freqs = 62
        sigma = 1
        ndatapoints = 2000

        outcomes = run_evaluation(ndatapoints, kernel_width, aa_dim, num_aas,
                    num_freqs, sigma, mode = "maxpool")
        for outcome in outcomes:
            self.assertTrue(outcome)
        outcomes = run_evaluation(ndatapoints, kernel_width, aa_dim, num_aas,
                    num_freqs, sigma, mode = "maxpool", precision = "float")
        for outcome in outcomes:
            self.assertTrue(outcome)


    def test_conv1d_maxpool_loc(self):
        """Tests the C / Cuda FHT-based convolution with global max pooling
        functions and with adjustment based on global mean pooling."""
        kernel_width = 9
        num_aas = 23
        aa_dim = 21
        num_freqs = 128
        sigma = 1
        ndatapoints = 124
        outcomes = run_evaluation(ndatapoints, kernel_width, aa_dim, num_aas,
                    num_freqs, sigma, mode = "maxpool_loc")
        for outcome in outcomes:
            self.assertTrue(outcome)

        kernel_width = 5
        num_aas = 56
        aa_dim = 2
        num_freqs = 62
        sigma = 1
        ndatapoints = 2000

        outcomes = run_evaluation(ndatapoints, kernel_width, aa_dim, num_aas,
                    num_freqs, sigma, mode = "maxpool_loc")
        for outcome in outcomes:
            self.assertTrue(outcome)
        outcomes = run_evaluation(ndatapoints, kernel_width, aa_dim, num_aas,
                    num_freqs, sigma, mode = "maxpool_loc", precision = "float")
        for outcome in outcomes:
            self.assertTrue(outcome)


def run_evaluation(ndatapoints, kernel_width, aa_dim, num_aas,
                    num_freqs, sigma, mode, precision = "double"):
    """Rungs an evaluation for the FHT-based functions using some
    set of input settings (e.g. do or do not calculate gradient,
    use CPU or GPU, use X number of frequencies etc."""
    gradient, true_gradient = None, None
    dim2, num_blocks, xdata, reshaped_x, features, s_mat, \
                radem = get_initial_matrices_fht(ndatapoints, kernel_width,
                        aa_dim, num_aas, num_freqs, mode, precision)
    if mode.startswith("maxpool"):
        true_features = get_features_maxpool(xdata, kernel_width, dim2,
                            radem, s_mat, num_freqs, num_blocks, mode,
                            precision)
    elif mode == "conv":
        true_features = get_features(xdata, kernel_width, dim2,
                            radem, s_mat, num_freqs, num_blocks, sigma,
                            precision)
    elif mode == "conv_gradient":
        true_features, true_gradient = get_features_with_gradient(xdata,
                            kernel_width, dim2, radem, s_mat, num_freqs,
                            num_blocks, sigma, precision)
    if precision == "float":
        conv_fun = floatCpuConv1dTransform
    else:
        conv_fun = doubleCpuConv1dTransform

    if "gradient" not in mode:
        conv_fun(reshaped_x, radem, features, s_mat,
                2, sigma, mode)
        if "maxpool" in mode:
            features = features[:,:num_freqs]
        else:
            features = features[:,:(2 * num_freqs)]
        outcome_gradient = True
    else:
        gradient = conv_fun(reshaped_x, radem, features, s_mat,
                2, sigma, mode)
        features = features[:,:(2*num_freqs)]
        gradient = gradient[:,:(2*num_freqs)]
    
    if precision == "float":
        outcome = np.allclose(features, true_features, rtol=1e-3, atol=1e-3)
    else:
        outcome = np.allclose(features, true_features)
    print(f"Settings: N {ndatapoints}, kernel_width {kernel_width}, "
        f"aa_dim: {aa_dim}, num_aas: {num_aas}, num_freqs: {num_freqs}, "
        f"sigma: {sigma}, mode: {mode}, precision {precision}\n"
        f"Does result match on CPU? {outcome}")

    if gradient is not None:
        if precision == "float":
            outcome_gradient = np.allclose(features, true_features, rtol=1e-4, atol=1e-4)
        else:
            outcome_gradient = np.allclose(features, true_features)
        print(f"Settings: N {ndatapoints}, kernel_width {kernel_width}, "
            f"aa_dim: {aa_dim}, num_aas: {num_aas}, num_freqs: {num_freqs}, "
            f"sigma: {sigma}, mode: {mode}, precision: {precision}\n"
            f"Does gradient match on CPU? {outcome_gradient}")


    if "cupy" not in sys.modules:
        return outcome, outcome_gradient

    dim2, num_blocks, xdata, reshaped_x, features, s_mat, \
                radem = get_initial_matrices_fht(ndatapoints, kernel_width,
                        aa_dim, num_aas, num_freqs, mode, precision)
    if precision == "float":
        conv_fun = floatGpuConv1dTransform
    else:
        conv_fun = doubleGpuConv1dTransform
    xdata = cp.asarray(xdata)
    reshaped_x = cp.asarray(reshaped_x)
    features = cp.asarray(features)
    s_mat = cp.asarray(s_mat)
    radem = cp.asarray(radem)

    if "gradient" not in mode:
        conv_fun(reshaped_x, radem, features, s_mat,
                2, sigma, mode)
        if "maxpool" in mode:
            features = features[:,:num_freqs]
        else:
            features = features[:,:(2 * num_freqs)]
        outcome_cuda_gradient = True
    else:
        gradient = conv_fun(reshaped_x, radem, features, s_mat,
                2, sigma, mode)
        features = features[:,:(2*num_freqs)]
        gradient = cp.asnumpy(gradient[:,:(2*num_freqs)])
    
    features = cp.asnumpy(features)
    
    if precision == "float":
        outcome_cuda = np.allclose(features, true_features, rtol=1e-4, atol=1e-4)
    else:
        outcome_cuda = np.allclose(features, true_features)
    print(f"Settings: N {ndatapoints}, kernel_width {kernel_width}, "
        f"aa_dim: {aa_dim}, num_aas: {num_aas}, num_freqs: {num_freqs}, "
        f"sigma: {sigma}, mode: {mode}, precision {precision}\n"
        f"Does result match on cuda? {outcome_cuda}")

    if gradient is not None:
        if precision == "float":
            outcome_cuda_gradient = np.allclose(features, true_features,
                    rtol=1e-3, atol=1e-3)
        else:
            outcome_cuda_gradient = np.allclose(features, true_features)
        print(f"Settings: N {ndatapoints}, kernel_width {kernel_width}, "
            f"aa_dim: {aa_dim}, num_aas: {num_aas}, num_freqs: {num_freqs}, "
            f"sigma: {sigma}, mode: {mode}, precision: {precision}\n"
            f"Does gradient match on cuda? {outcome_cuda_gradient}")

    print("\n")
    return outcome, outcome_gradient, outcome_cuda, outcome_cuda_gradient





if __name__ == "__main__":
    unittest.main()
