"""Checks the matmul based convolution
feature generation routines to ensure they are producing
correct results by comparing with a clunky, simple
mostly Python implementation."""
import sys

import unittest
import numpy as np

from kernel_tools import CPUConv1dMMTransform
try:
    import cupy as cp
    from kernel_tools import GPUConv1dMMTransform
except:
    pass


class TestConv1dMM(unittest.TestCase):
    """Tests matmul-based convolution feature generation."""

    def test_conv1d(self):
        """Tests the matmul-based Conv1d Cython functions."""
        kernel_width = 9
        num_aas = 23
        aa_dim = 21
        num_freqs = 1000
        sigma = 1
        ndatapoints = 124

        features, true_features, _, _ = run_evaluation(ndatapoints, kernel_width, aa_dim, num_aas,
                    num_freqs, sigma, device = "cpu")
        outcome = np.allclose(true_features, features)
        print(f"Does result match for ndatapoints={ndatapoints}, num_aas={num_aas}, "
            f"kernel_width={kernel_width} on CPU? {outcome}")
        self.assertTrue(outcome)

        if "cupy" in sys.modules:
            features, true_features, _, _ = run_evaluation(ndatapoints, kernel_width,
                    aa_dim, num_aas, num_freqs, sigma, device = "gpu")
            outcome = np.max(np.abs(true_features - features)) < 2e-5
            print(f"Does result match for ndatapoints={ndatapoints}, num_aas={num_aas}, "
                f"kernel_width={kernel_width} on CUDA? {outcome}")
            self.assertTrue(outcome)
        kernel_width = 5
        num_aas = 56
        aa_dim = 2
        num_freqs = 62
        sigma = 1
        ndatapoints = 2000

        features, true_features, _, _ = run_evaluation(ndatapoints, kernel_width, aa_dim, num_aas,
                    num_freqs, sigma)
        outcome = np.allclose(true_features, features)
        print(f"Does result match for ndatapoints={ndatapoints}, num_aas={num_aas}, "
            f"kernel_width={kernel_width} {outcome}")
        self.assertTrue(outcome)
        if "cupy" in sys.modules:
            features, true_features, _, _ = run_evaluation(ndatapoints, kernel_width,
                    aa_dim, num_aas, num_freqs, sigma, device = "gpu")
            outcome = np.max(np.abs(true_features - features)) < 1e-5
            print(f"Does result match for ndatapoints={ndatapoints}, num_aas={num_aas}, "
                f"kernel_width={kernel_width} on CUDA? {outcome}")
            self.assertTrue(outcome)





    def test_conv1d_gradients(self):
        """Tests the gradient calculations for the matmul-based Cython
        functions."""
        kernel_width = 9
        num_aas = 23
        aa_dim = 21
        num_freqs = 128
        sigma = 1
        ndatapoints = 36

        _, _, gradient, true_grad = \
                run_evaluation(ndatapoints, kernel_width, aa_dim, num_aas,
                    num_freqs, sigma, get_gradient = True, device = "cpu")
        outcome = np.allclose(true_grad, gradient)

        print(f"Does gradient match for ndatapoints={ndatapoints}, num_aas={num_aas}, "
                f"kernel_width={kernel_width} on CPU? {outcome}")
        self.assertTrue(outcome)
        if "cupy" in sys.modules:
            _, _, ggradient, gtrue_grad = \
                run_evaluation(ndatapoints, kernel_width, aa_dim, num_aas,
                    num_freqs, sigma, get_gradient = True, device = "gpu")
            outcome = np.max(np.abs(true_grad - gradient)) < 1e-5
            print(f"Does gradient match for ndatapoints={ndatapoints}, num_aas={num_aas}, "
                f"kernel_width={kernel_width} on CUDA? {outcome}")
            self.assertTrue(outcome)

        kernel_width = 5
        num_aas = 56
        aa_dim = 2
        num_freqs = 62
        sigma = 1
        ndatapoints = 1

        _, _, gradient, true_grad = \
                run_evaluation(ndatapoints, kernel_width, aa_dim, num_aas,
                    num_freqs, sigma, get_gradient = True)
        outcome = np.allclose(true_grad, gradient)
        print(f"Does gradient match for ndatapoints={ndatapoints}, num_aas={num_aas}, "
                f"kernel_width={kernel_width}? {outcome}")
        self.assertTrue(outcome)
        if "cupy" in sys.modules:
            _, _, gradient, true_grad = \
                run_evaluation(ndatapoints, kernel_width, aa_dim, num_aas,
                    num_freqs, sigma, get_gradient = True, device = "gpu")
            outcome = np.max(np.abs(true_grad - gradient)) < 1e-5
            print(f"Does gradient match for ndatapoints={ndatapoints}, num_aas={num_aas}, "
                f"kernel_width={kernel_width} on CUDA? {outcome}")
            self.assertTrue(outcome)


    def test_conv1d_maxpool(self):
        """Tests the Cython matmul-based convolution with global max pooling
        functions."""
        kernel_width = 9
        num_aas = 23
        aa_dim = 21
        num_freqs = 128
        sigma = 1
        ndatapoints = 124
        features, true_features, _, _ = run_evaluation(ndatapoints, kernel_width, aa_dim, num_aas,
                    num_freqs, sigma, max_pool = True, device = "cpu")
        outcome = np.allclose(true_features, features)
        print(f"Does maxpool result match for ndatapoints={ndatapoints}, num_aas={num_aas}, "
                f"kernel_width={kernel_width} on CPU? {outcome}")
        self.assertTrue(outcome)
        if "cupy" in sys.modules:
            features, true_features, _, _ = run_evaluation(ndatapoints, kernel_width,
                    aa_dim, num_aas, num_freqs, sigma, max_pool = True,
                    device = "gpu")
            outcome = np.allclose(true_features, features)
            print(f"Does maxpool result match for ndatapoints={ndatapoints}, num_aas={num_aas}, "
                f"kernel_width={kernel_width} on CUDA? {outcome}")
            self.assertTrue(outcome)

        kernel_width = 5
        num_aas = 56
        aa_dim = 2
        num_freqs = 62
        sigma = 1
        ndatapoints = 2000

        features, true_features, _, _ = run_evaluation(ndatapoints, kernel_width, aa_dim, num_aas,
                    num_freqs, sigma, max_pool = True, device = "cpu")
        outcome = np.allclose(true_features, features)
        print(f"Does maxpool result match for ndatapoints={ndatapoints}, num_aas={num_aas}, "
                f"kernel_width={kernel_width} on CPU? {outcome}")
        self.assertTrue(outcome)
        if "cupy" in sys.modules:
            features, true_features, _, _ = run_evaluation(ndatapoints, kernel_width,
                    aa_dim, num_aas, num_freqs, sigma, max_pool = True,
                    device = "gpu")
            outcome = np.allclose(true_features, features)
            print(f"Does maxpool result match for ndatapoints={ndatapoints}, num_aas={num_aas}, "
                f"kernel_width={kernel_width} on CUDA? {outcome}")
            self.assertTrue(outcome)


def run_evaluation(ndatapoints, kernel_width, aa_dim, num_aas,
                    num_freqs, sigma, get_gradient = False,
                    max_pool = False, device = "cpu"):
    """Rungs an evaluation for the matmul-based functions using some
    set of input settings (e.g. do or do not calculate gradient,
    use CPU or GPU, use X number of frequencies etc."""
    gradient, true_gradient = None, None
    xdata, features, kernels = get_initial_matrices_mm(ndatapoints, kernel_width,
                            aa_dim, num_aas, num_freqs, max_pool)
    if max_pool:
        true_features = get_features_maxpool(xdata, kernels,
                            kernel_width)
    elif not get_gradient:
        true_features = get_features(xdata, kernels, kernel_width,
                            sigma)
    else:
        true_features, true_gradient = get_features_with_gradient(xdata,
                                kernels, kernel_width, sigma, device)
    if device == "cpu":
        conv_fun = CPUConv1dMMTransform
    else:
        conv_fun = GPUConv1dMMTransform
        xdata = cp.asarray(xdata).astype(cp.float32)
        kernels = cp.asarray(kernels).astype(cp.float32)
        features = cp.asarray(features).astype(cp.float64)
    if max_pool:
        conv_fun(xdata, kernels, features, sigma, mode = "maxpool")
    elif not get_gradient:
        conv_fun(xdata, kernels, features, sigma, mode = "conv")
    else:
        gradient = conv_fun(xdata, kernels, features, sigma,
                mode = "conv_grad")
 
    if not get_gradient:
        if device == "gpu":
            features = cp.asnumpy(features)
    if device == "gpu":
        features = cp.asnumpy(features)
        gradient = cp.asnumpy(gradient)
    return features, true_features, gradient, true_gradient


def get_initial_matrices_mm(ndatapoints, kernel_width, aa_dim, num_aas,
            num_freqs, max_pool):
    """Supplies the initial matrices for the matmul tests."""
    dim2 = aa_dim * kernel_width

    random_seed = 123
    rng = np.random.default_rng(random_seed)
    xdata = rng.uniform(low=-10.0,high=10.0, size=(ndatapoints, num_aas, aa_dim))

    if not max_pool:
        features = np.zeros((ndatapoints, 2 * num_freqs))
    else:
        features = np.zeros((ndatapoints, num_freqs))
    kernels = rng.standard_normal(size = (num_freqs, dim2))
    return xdata, features, kernels



def get_features_maxpool(xdata, kernels, kernel_width):
    """Simple straightforward feature generation with global
    max pooling, no activation."""
    reshaped_x = get_reshaped_x(xdata, kernel_width)
    features = np.zeros((xdata.shape[0], kernels.shape[0]))
    chunk_size = 256
    for i in range(0, kernels.shape[0], chunk_size):
        x_temp = np.einsum("ij,mkj->mik", kernels[i:i+chunk_size,:],
                                reshaped_x)
        features[:,i:i+chunk_size] = x_temp.max(axis=2)
    return features

def get_features(xdata, kernels, kernel_width, sigma):
    """Simple straightforward feature generation for the Conv1d
    kernel."""
    reshaped_x = get_reshaped_x(xdata, kernel_width)
    chunk_size = min(256, kernels.shape[0])
    scaling_factor = np.sqrt(1 / float(kernels.shape[0]))
    features = np.zeros((xdata.shape[0], 2 * kernels.shape[0]))

    for i in range(0, kernels.shape[0], chunk_size):
        x_temp = np.einsum("ij,mkj->mik", kernels[i:i+chunk_size,:],
                                reshaped_x)
        klow, khigh = i, min(i + chunk_size, kernels.shape[0])
        features[:,klow:khigh] = np.cos(sigma * x_temp).sum(axis=2)
        flow, fhigh = kernels.shape[0] + i, kernels.shape[0] + khigh
        features[:,flow:fhigh] = np.sin(sigma * x_temp).sum(axis=2)
    features *= scaling_factor
    return features


def get_features_with_gradient(xdata, kernels, kernel_width, sigma, device = "cpu"):
    """Simple straightforward feature generation for the Conv1d
    kernel, with gradients."""
    reshaped_x = get_reshaped_x(xdata, kernel_width)
    chunk_size = 256
    scaling_factor = np.sqrt(1 / float(kernels.shape[0]))
    features = np.zeros((xdata.shape[0], 2 * kernels.shape[0]))
    gradient = np.zeros(features.shape)
    if device == "gpu":
        features = cp.asarray(features).astype(cp.float32)
        gradient = cp.asarray(gradient).astype(cp.float32)
        kernels = cp.asarray(kernels).astype(cp.float32)
        reshaped_x = cp.asarray(reshaped_x).astype(cp.float32)

    for i in range(0, kernels.shape[0], chunk_size):
        klow, khigh = i, min(i + chunk_size, kernels.shape[0])
        flow, fhigh = kernels.shape[0] + i, kernels.shape[0] + i + chunk_size
        if device == "cpu":
            x_temp = np.einsum("ij,mkj->mik", kernels[i:i+chunk_size,:],
                                reshaped_x)
            features[:,klow:khigh] = np.cos(sigma * x_temp).sum(axis=2)
            features[:,flow:fhigh] = np.sin(sigma * x_temp).sum(axis=2)

            gradient[:,klow:khigh] = (-np.sin(sigma * x_temp) * x_temp).sum(axis=2)
            gradient[:,flow:fhigh] = (np.cos(sigma * x_temp) * x_temp).sum(axis=2)
        else:
            x_temp = cp.einsum("ij,mkj->mik", kernels[i:i+chunk_size,:],
                                reshaped_x)
            features[:,klow:khigh] = cp.cos(sigma * x_temp).sum(axis=2)
            features[:,flow:fhigh] = cp.sin(sigma * x_temp).sum(axis=2)

            gradient[:,klow:khigh] = (-cp.sin(sigma * x_temp) * x_temp).sum(axis=2)
            gradient[:,flow:fhigh] = (cp.cos(sigma * x_temp) * x_temp).sum(axis=2)

    features *= scaling_factor
    gradient *= scaling_factor
    if device == "gpu":
        features = cp.asnumpy(features).astype(np.float64)
        kernels = cp.asnumpy(kernels).astype(np.float64)
        gradient = cp.asnumpy(gradient).astype(np.float64)
        reshaped_x = cp.asnumpy(reshaped_x).astype(np.float64)
    return features, gradient


def get_reshaped_x(xdata, kernel_width):
    """This is an extremely inefficient way to reshape an array
    for a convolution but used here to double-check the procedures
    performed by the Cython funcs."""
    dim2 = xdata.shape[2] * kernel_width
    num_blocks = xdata.shape[1] - kernel_width + 1
    reshaped_x = np.zeros((xdata.shape[0], num_blocks, dim2))
    for i in range(xdata.shape[1] - kernel_width + 1):
        window = xdata[:,i:i+kernel_width,:]
        reshaped_x[:,i,:] = window.reshape((window.shape[0],
                        window.shape[1] * window.shape[2]))
    return reshaped_x


if __name__ == "__main__":
    unittest.main()
