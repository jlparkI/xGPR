"""Provides tools for implementing convolutions using either
ReLU with global max pooling or average pooling with cosine
activation."""
import numpy as np
cimport numpy as np
try:
    import cupy as cp
    from cupyx.scipy.signal import fftconvolve as GPUConvolve
    from cupyx.scipy.ndimage import convolve as GPUImageConvolve
except:
    pass
from scipy.signal import fftconvolve as CPUConvolve




def GPUConv1dMMTransform(X, kernels, features,
                double sigma, str mode = "conv"):
    """Implements the transform for the Conv1d kernel using matrix
    multiplication, in order to compare speed with the FHT implementation.

    Args:
        X (cp.ndarray): The raw input data. Should be a 3d array.
        kernels (cp.ndarray): An array of shape (N, M, D)
            for N random features, M kernel width and D input
            features per timepoint.
        features (cp.ndarray): An empty array in which
            the results will be stored. shape[0] must == shape[0]
            of X, shape[1] must == 2 * shape[0] of kernels.
        sigma (double): A kernel-specific hyperparameter that
            determines the mismatch tolerance for the Ngram kernel.
        mode (str): One of 'conv', 'conv_grad', 'maxpool', or
            'maxpool_loc'. Determines whether we produce 1) the
            Conv1d kernel (approximates Ngram kernel), 2) the
            Conv1d kernel with gradient, 3) maxpool feature extraction,
            4) maxpool feature extraction with ratio info

    Returns:
        Nothing unless get_gradient is True. The random features
            are stored in the features input array.
        gradient (cp.ndarray): An array containing the gradient.
    """
    cdef int i, nfeatures, kernel_size, num_slides
    cdef klow, khigh, flow, fhigh
    cdef int conv_width
    cdef double scaling_factor
    #256 is a chunk size to prevent us from using too much memory
    #at once (if we simply batch-matrix-multiply, memory consumption
    #will be astronomical for large numbers of kernels).
    cdef int chunk_size = 256

    if len(X.shape) != 3 or len(kernels.shape) != 2 or len(features.shape) != 2:
        raise ValueError("The dimensions of one or more input arrays to GPUConvNgramTransform"
                " are incorrect.")
    if X.dtype != "float32" or kernels.dtype != "float32" or features.dtype != "float64":
        raise ValueError("For the GPUConv1dMMTransform, all input arrays "
                "must be float32, except for features which is float64.")
    if not X.flags["C_CONTIGUOUS"]:
        raise ValueError("The x-array input to GPUConv1dMMTransform is not c-contiguous.")
    if not kernels.flags["C_CONTIGUOUS"]:
        raise ValueError("The kernels input to GPUConv1dMMTransform is not c-contiguous.")
    if not features.flags["C_CONTIGUOUS"]:
        raise ValueError("The features input to GPUConv1dMMTransform is not c-contiguous.")

    nfeatures = kernels.shape[0]
    kernel_size = kernels.shape[1]
    if kernel_size % X.shape[2] != 0:
        raise ValueError("Kernel size must be an integer multiple of "
                "X.shape[2].")
    conv_width = kernel_size // X.shape[2]
    if conv_width >= X.shape[1]:
        raise ValueError("conv_width must be < X.shape[1].")

    num_slides = X.shape[1] - conv_width + 1
    scaling_factor = np.sqrt(1 / <float>nfeatures)

    #For non-max-pool, we generate 2 features per filter. For
    #max pool, we generate 1.
    if 2 * nfeatures != features.shape[1] and "maxpool" not in mode:
        raise ValueError("The number of features and size of the filter matrix "
                "does not match.")
    elif nfeatures != features.shape[1] and "maxpool" in mode:
        raise ValueError("The number of features and size of the filter matrix "
                "does not match.")
    #Stride tricks is a little dangerous. TODO: Find efficient non stride-tricks
    #way to implement this restructuring.
    x_strided = cp.lib.stride_tricks.as_strided(X, shape=(X.shape[0],
                            num_slides, kernel_size),
                            strides=(X.strides[0], X.shape[2] *
                                X.strides[2], X.strides[2]))
    if mode == "maxpool":
        for i in range(0, kernels.shape[0], chunk_size):
            klow, khigh = i, min(i + chunk_size, nfeatures)
            x_temp = cp.einsum("ij,mkj->mik",
                    kernels[klow:khigh, :], x_strided)
            features[:, klow:khigh] = x_temp.max(axis=2)
    elif mode == "maxpool_loc":
        for i in range(0, kernels.shape[0], chunk_size):
            klow, khigh = i, min(i + chunk_size, nfeatures)
            x_temp = cp.einsum("ij,mkj->mik", kernels[klow:khigh, :], x_strided)
            features[:, klow:khigh] = x_temp.max(axis=2) - x_temp.mean(axis=2)
    elif mode == "max_std":
        for i in range(0, kernels.shape[0], chunk_size):
            klow, khigh = i, min(i + chunk_size, nfeatures)
            flow, fhigh = nfeatures + klow, nfeatures + khigh
            x_temp = np.einsum("ij,mkj->mik", kernels[klow:khigh, :], x_strided)
            features[:, klow:khigh] = x_temp.max(axis=2) - x_temp.mean(axis=2)
            features[:, flow:fhigh] = x_temp.std(axis=2)
    elif mode == "twolayer":
        num_slides = x_strided.shape[1] - conv_width + 1
        for i in range(0, kernels.shape[0], chunk_size):
            klow, khigh = i, min(i + chunk_size, nfeatures)
            flow, fhigh = nfeatures + klow, nfeatures + khigh
            x_temp = cp.einsum("ij,mkj->mki", kernels[klow:khigh, :], x_strided)
            features[:, klow:khigh] = x_temp.max(axis=1) - x_temp.mean(axis=1)
            #Second layer. See TODO for stride tricks above.
            x_temp = cp.ascontiguousarray(x_temp[:,:,:X.shape[2]])
            x_temp[:] = cp.clip(x_temp, a_min = 0, a_max = None, out = x_temp)
            x_temp = cp.lib.stride_tricks.as_strided(x_temp, shape=(x_temp.shape[0],
                            num_slides, x_temp.shape[2] * conv_width),
                            strides=(x_temp.strides[0], x_temp.shape[2] *
                                x_temp.strides[2], x_temp.strides[2]))
            x_temp = cp.einsum("ij,mkj->mik", kernels[klow:khigh, :], x_temp)
            features[:, flow:fhigh] = x_temp.max(axis=2) - x_temp.mean(axis=2)

    elif mode == "conv":
        for i in range(0, kernels.shape[0], chunk_size):
            klow, khigh = i, min(i + chunk_size, nfeatures)
            flow, fhigh = nfeatures + klow, nfeatures + khigh
            x_temp = sigma * cp.einsum("ij,mkj->mik",
                    kernels[klow:khigh, :], x_strided)
            features[:, klow:khigh] = cp.cos(x_temp).sum(axis=2)
            features[:, flow:fhigh] = cp.sin(x_temp).sum(axis=2)
        features *= scaling_factor
    elif mode == "conv_grad":
        gradient = cp.empty((features.shape[0], features.shape[1]),
                        dtype = cp.float32)
        for i in range(0, kernels.shape[0], chunk_size):
            klow, khigh = i, min(i + chunk_size, nfeatures)
            flow, fhigh = nfeatures + klow, nfeatures + khigh
            x_temp = cp.einsum("ij,mkj->mik",
                    kernels[klow:khigh, :], x_strided)
            features[:, klow:khigh] = cp.cos(sigma * x_temp).sum(axis=2)
            gradient[:, klow:khigh] = (-cp.sin(sigma * x_temp) * 
                                x_temp).sum(axis=2)
            features[:, flow:fhigh] = cp.sin(sigma * x_temp).sum(axis=2)
            gradient[:, flow:fhigh] = (cp.cos(sigma * x_temp) * 
                                x_temp).sum(axis=2)

        features *= scaling_factor
        gradient *= scaling_factor
        return gradient
    else:
        raise ValueError("Unrecognized mode supplied to a kernel_tools function.")


def CPUConv1dMMTransform(np.ndarray[np.float64_t, ndim=3] X,
                np.ndarray[np.float64_t, ndim=2] kernels,
                np.ndarray[np.float64_t, ndim=2] features,
                double sigma, str mode):
    """Implements the transform for the Conv1d kernel using matrix
    multiplication, in order to compare speed with the FHT implementation.

    Args:
        X (np.ndarray): The raw input data. Should be a 3d array.
        kernels (np.ndarray): A numpy array of shape (N, M, D)
            for N random features, M kernel width and D input
            features per timepoint.
        features (np.ndarray): An empty numpy array in which
            the results will be stored. shape[0] must == shape[0]
            of X, shape[1] must == 2 * shape[0] of kernels.
        sigma (double): A kernel-specific hyperparameter that
            determines the mismatch tolerance for the Ngram kernel.
        mode (str): One of 'conv', 'conv_grad', 'maxpool', or
            'maxpool_loc'. Determines whether we produce 1) the
            Conv1d kernel (approximates Ngram kernel), 2) the
            Conv1d kernel with gradient, 3) maxpool feature extraction,
            4) maxpool feature extraction with location info
            (most useful for sequences that can be aligned).

    Returns:
        Nothing unless get_gradient is True. The random features
            are stored in the features input array.
        gradient (np.ndarray): A numpy array containing the gradient.
    """
    cdef int i, nfeatures, kernel_size, num_slides
    cdef klow, khigh, flow, fhigh
    cdef int conv_width
    cdef double scaling_factor
    cdef np.ndarray[np.float64_t, ndim=2] gradient
    cdef np.ndarray[np.float64_t, ndim=3] x_temp
    cdef np.ndarray[np.float64_t, ndim=3] x_strided
    
    #256 is a chunk size to prevent us from using too much memory
    #at once (if we simply batch-matrix-multiply, memory consumption
    #will be astronomical for large numbers of kernels).
    cdef int chunk_size = 256

    nfeatures = kernels.shape[0]
    kernel_size = kernels.shape[1]
    if kernel_size % X.shape[2] != 0:
        raise ValueError("Kernel size must be an integer multiple of "
                "X.shape[2].")
    conv_width = kernel_size // X.shape[2]
    if conv_width >= X.shape[1]:
        raise ValueError("conv_width must be < X.shape[1].")

    num_slides = X.shape[1] - conv_width + 1
    scaling_factor = np.sqrt(1 / <float>nfeatures)
    #For non-max-pool, we generate 2 features per filter. For
    #max pool, we generate 1.
    if 2 * nfeatures != features.shape[1] and "maxpool" not in mode:
        raise ValueError("The number of features and size of the filter matrix "
                "does not match.")
    elif nfeatures != features.shape[1] and "maxpool" in mode:
        raise ValueError("The number of features and size of the filter matrix "
                "does not match.")
    if not X.flags["C_CONTIGUOUS"]:
        raise ValueError("The x-array input to CPUConv1dMMTransform is not c-contiguous.")
    if not kernels.flags["C_CONTIGUOUS"]:
        raise ValueError("The kernels input to CPUConv1dMMTransform is not c-contiguous.")
    if not features.flags["C_CONTIGUOUS"]:
        raise ValueError("The features input to CPUConv1dMMTransform is not c-contiguous.")

    #Stride tricks is a little dangerous. TODO: Find efficient non stride-tricks
    #way to implement this restructuring.
    x_strided = np.lib.stride_tricks.as_strided(X, shape=(X.shape[0],
                            num_slides, kernel_size),
                            strides=(X.strides[0], X.shape[2] *
                                X.strides[2], X.strides[2]))
    if mode == "maxpool":
        for i in range(0, kernels.shape[0], chunk_size):
            klow, khigh = i, min(i + chunk_size, nfeatures)
            x_temp = np.einsum("ij,mkj->mik",
                    kernels[klow:khigh, :], x_strided)
            features[:, klow:khigh] = x_temp.max(axis=2)
    elif mode == "maxpool_loc":
        for i in range(0, kernels.shape[0], chunk_size):
            klow, khigh = i, min(i + chunk_size, nfeatures)
            x_temp = np.einsum("ij,mkj->mik",
                    kernels[klow:khigh, :], x_strided)
            features[:, klow:khigh] = x_temp.max(axis=2) - x_temp.mean(axis=2)
    elif mode == "max_std":
        for i in range(0, kernels.shape[0], chunk_size):
            klow, khigh = i, min(i + chunk_size, nfeatures)
            flow, fhigh = nfeatures + klow, nfeatures + khigh
            x_temp = sigma * np.einsum("ij,mkj->mik",
                    kernels[klow:khigh, :], x_strided)
            features[:, klow:khigh] = x_temp.max(axis=2) - x_temp.mean(axis=2)
            features[:, flow:fhigh] = x_temp.std(axis=2)
    elif mode == "twolayer":
        num_slides = x_strided.shape[1] - conv_width + 1
        for i in range(0, kernels.shape[0], chunk_size):
            klow, khigh = i, min(i + chunk_size, nfeatures)
            flow, fhigh = nfeatures + klow, nfeatures + khigh
            x_temp = np.einsum("ij,mkj->mki", kernels[klow:khigh, :], x_strided)
            features[:, klow:khigh] = x_temp.max(axis=1) - x_temp.mean(axis=1)
            #Second layer. See TODO for stride tricks above.
            x_temp = np.ascontiguousarray(x_temp[:,:,:X.shape[2]])
            x_temp[:] = np.clip(x_temp, a_min = 0, a_max = None, out = x_temp)
            x_temp = np.lib.stride_tricks.as_strided(x_temp, shape=(x_temp.shape[0],
                            num_slides, x_temp.shape[2] * conv_width),
                            strides=(x_temp.strides[0], x_temp.shape[2] *
                                x_temp.strides[2], x_temp.strides[2]))
            x_temp = np.einsum("ij,mkj->mik", kernels[klow:khigh, :], x_temp)
            features[:, flow:fhigh] = x_temp.max(axis=2) - x_temp.mean(axis=2)

    elif mode == "conv":
        for i in range(0, kernels.shape[0], chunk_size):
            klow, khigh = i, min(i + chunk_size, nfeatures)
            flow, fhigh = nfeatures + klow, nfeatures + khigh
            x_temp = sigma * np.einsum("ij,mkj->mik",
                    kernels[klow:khigh, :], x_strided)
            features[:, klow:khigh] = np.cos(x_temp).sum(axis=2)
            features[:, flow:fhigh] = np.sin(x_temp).sum(axis=2)
        features *= scaling_factor
    elif mode == "conv_grad":
        gradient = np.empty((features.shape[0], features.shape[1]))
        for i in range(0, kernels.shape[0], chunk_size):
            klow, khigh = i, min(i + chunk_size, nfeatures)
            flow, fhigh = nfeatures + klow, nfeatures + khigh
            x_temp = np.einsum("ij,mkj->mik",
                    kernels[klow:khigh, :], x_strided)
            features[:, klow:khigh] = np.cos(sigma * x_temp).sum(axis=2)
            gradient[:, klow:khigh] = (-np.sin(sigma * x_temp) * 
                                x_temp).sum(axis=2)
            features[:, flow:fhigh] = np.sin(sigma * x_temp).sum(axis=2)
            gradient[:, flow:fhigh] = (np.cos(sigma * x_temp) * 
                                x_temp).sum(axis=2)

        features *= scaling_factor
        gradient *= scaling_factor
        return gradient
    else:
        raise ValueError("Unrecognized mode supplied to a kernel_tools function.")
