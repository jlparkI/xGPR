"""Handles convolution-type hadamard transform based operations for graphs
and sequences if the input array is an array of floats.

Also performs all of the bounds and safety checks needed to use these
functions (the C functions do not do their own bounds checking). It
is EXTREMELY important that this wrapper not be bypassed for this
reason -- it double checks all of the array dimensions, types,
is data contiguous etc. before calling the wrapped C functions."""
import numpy as np
cimport numpy as np
cimport cython
from libc cimport stdint
import cupy as cp
from libc.stdint cimport uintptr_t
import math
from libc.stdint cimport int8_t, int32_t
from libcpp cimport bool


cdef extern from "convolution_ops/convolution.h" nogil:
    const char *conv1dPrep[T](int8_t *radem,
                T reshapedx[], int reshapeddim0, 
                int reshapeddim1, int reshapeddim2,
                int startposition, int numfreqs)

cdef extern from "convolution_ops/rbf_convolution.h" nogil:
    const char *convRBFFeatureGen[T](const int8_t *radem, const T xdata[],
            const T chiArr[], double *outputArray, int32_t *seqlengths,
            int xdim0, int xdim1, int xdim2, int numFreqs,
            int rademShape2, int convWidth, int paddedBufferSize,
            double scalingTerm)
    const char *convRBFFeatureGrad[T](const int8_t *radem, const T xdata[],
            const T chiArr[], double *outputArray, int32_t *seqlengths,
            double *gradientArray, double sigma,
            int xdim0, int xdim1, int xdim2, int numFreqs,
            int rademShape2, int convWidth, int paddedBufferSize,
            double scalingTerm)


@cython.boundscheck(False)
@cython.wraparound(False)
def gpuConv1dMaxpool(reshapedX, radem, outputArray, chiArr,
        int numThreads):
    """Uses wrapped C extensions to perform random feature generation
    with ReLU activation and maxpooling. TODO: Transfer the loop
    and sum operations in here to a Cuda kernel and wrap.

    Args:
        reshapedX (cp.ndarray): An array of type float32 from which
            the features will be generated. Is not modified. Must
            be of shape (N x D x C) where C is a power of 2. Should
            have been reshaped to be appropriate for convolution.
        radem (cp.ndarray): A stack of diagonal matrices of type int8_t
            of shape (3 x 1 x m * C), where R is the number of random
            features requested and m is ceil(R / C).
        outputArray (cp.ndarray): An N x R array in which the output features
            will be stored.
        chiArr (cp.ndarray): A stack of diagonal matrices stored as an
            array of shape (R) drawn from a chi distribution.
        num_threads (int): This argument is so that this function has
            the same interface as the CPU SORF Transform. It is not
            needed for the GPU transform and is ignored.
    """
    cdef const char *errCode
    cdef int i, startPosition, cutoff
    cdef int num_repeats = (radem.shape[2] + reshapedX.shape[2] - 1) // reshapedX.shape[2]
    reshapedXCopy = reshapedX.copy()
    cdef uintptr_t addr_reshapedCopy = reshapedXCopy.data.ptr
    cdef uintptr_t addr_radem = radem.data.ptr

    
    #Check that all arrays have expected sizes and data types.    
    if reshapedX.shape[0] == 0:
        raise ValueError("There must be at least one datapoint.")
    if reshapedX.shape[0] != outputArray.shape[0]:
        raise ValueError("The number of input and output datapoints do not "
                "agree.")
    if not len(radem.shape) == 3:
        raise ValueError("radem must be a 3d array.")
    if not len(chiArr.shape) == 1 or not len(outputArray.shape) == 2:
        raise ValueError("chiArr must be a 1d array; outputArray must be 2d.")
    if not len(reshapedX.shape) == 3:
        raise ValueError("X must be a 3d array.")
    if not radem.dtype == "int8":
        raise ValueError("radem must be int8.")


    #Check that shapes of radem, outputArray are correct.
    if radem.shape[0] != 3 or radem.shape[1] != 1:
        raise ValueError("radem must have length 3 for dim 0 and length 1 for dim1.")
    if outputArray.shape[1] != radem.shape[2]:
        raise ValueError("outputArray.shape[1] must be an integer multiple of the next largest "
                "power of 2 greater than the kernel width * X.shape[2].")

    #Next, make sure that reshapedX and chiArr make sense.
    if chiArr.shape[0] != radem.shape[2]:
        raise ValueError("chiArr.shape[0] must == radem.shape[2].")
        
    logdim = np.log2(reshapedX.shape[2])
    if np.ceil(logdim) != np.floor(logdim) or reshapedX.shape[2] < 2:
        raise ValueError("dim2 of the reshapedX array must be a power of 2 >= 2.")
    if not radem.shape[2] % reshapedX.shape[2] == 0:
        raise ValueError("The number of sampled frequencies should be an integer multiple of "
                "reshapedX.shape[2].")

    #Make sure that all inputs are C-contiguous.
    if not outputArray.flags["C_CONTIGUOUS"] or not reshapedX.flags["C_CONTIGUOUS"] \
            or not radem.flags["C_CONTIGUOUS"] or not chiArr.flags["C_CONTIGUOUS"]:
        raise ValueError("One or more arguments is not C contiguous.")


    
    startPosition, cutoff = 0, reshapedX.shape[2]
    
    if outputArray.dtype == "float64" and reshapedX.dtype == "float32" and \
            chiArr.dtype == "float32":
        for i in range(num_repeats):
            reshapedXCopy[:] = reshapedX
            errCode = conv1dPrep[float](<int8_t*>addr_radem,
                    <float*>addr_reshapedCopy, reshapedX.shape[0], reshapedX.shape[1], 
                    reshapedX.shape[2], i * reshapedX.shape[2],
                    radem.shape[2])
            if errCode.decode("UTF-8") != "no_error":
                raise Exception("Fatal error encountered in floatGpuConv1dTransform_.")
        
            reshapedXCopy *= chiArr[None,None,(i * reshapedX.shape[2]):((i+1) * reshapedX.shape[2])]
            outputArray[:,startPosition:cutoff] = reshapedXCopy.max(axis=1)

            cutoff += reshapedX.shape[2]
            startPosition += reshapedX.shape[2]

    elif outputArray.dtype == "float64" and reshapedX.dtype == "float64" and \
            chiArr.dtype == "float64":
        for i in range(num_repeats):
            reshapedXCopy[:] = reshapedX
            errCode = conv1dPrep[double](<int8_t*>addr_radem,
                    <double*>addr_reshapedCopy, reshapedX.shape[0], reshapedX.shape[1], 
                    reshapedX.shape[2], i * reshapedX.shape[2],
                    radem.shape[2])
            if errCode.decode("UTF-8") != "no_error":
                raise Exception("Fatal error encountered in floatGpuConv1dTransform_.")
        
            reshapedXCopy *= chiArr[None,None,(i * reshapedX.shape[2]):((i+1) * reshapedX.shape[2])]
            outputArray[:,startPosition:cutoff] = reshapedXCopy.max(axis=1)

            cutoff += reshapedX.shape[2]
            startPosition += reshapedX.shape[2]
    else:
        raise ValueError("Inconsistent types passed to a wrapped C++ function.")




@cython.boundscheck(False)
@cython.wraparound(False)
def gpuConv1dFGen(xdata, sequence_lengths, radem, outputArray,
                chiArr, int convWidth, int numThreads,
                bool averageFeatures = False):
    """Uses wrapped C functions to generate random features for Conv1d RBF-related kernels.
    This function cannot be used to calculate the gradient so is only used for forward pass
    only (during fitting, inference, non-gradient-based optimization). It does not multiply
    by the lengthscales, so caller should do this.

    Args:
        xdata (cp.ndarray): An input 3d raw data array of shape (N x D x C)
            for N datapoints. D must be >= convWidth.
        sequence_lengths (cp.ndarray): A 1d array of shape[0]==N where each element
            is the length of the corresponding sequence in xdata. All values must be
            >= convWidth.
        radem (cp.ndarray): A stack of diagonal matrices with elements drawn from the
            Rademacher distribution. Shape must be (3 x D x C).
        outputArray (cp.ndarray): A numpy array in which the generated features will be
            stored. Is modified in-place.
        chiArr (cp.ndarray): A stack of diagonal matrices stored as an
            array of shape m * C drawn from a chi distribution.
        convWidth (int): The width of the convolution. Must be <= D when xdata is (N x D x C).
        num_threads (int): Number of threads to use for FHT.
        averageFeatures (bool): Whether to average the features generated along the
            first axis (makes kernel result less dependent on sequence length / graph
            size).

    Raises:
        ValueError: A ValueError is raised if unexpected or invalid inputs are supplied.
    """
    cdef const char *errCode
    cdef double scalingTerm
    cdef double expectedNFreq
    cdef int paddedBufferSize

    if not isinstance(xdata, cp.ndarray) or not isinstance(sequence_lengths, cp.ndarray):
        raise ValueError("Passed non-cupy arrays to GPU function.")

    if len(chiArr.shape) != 1 or len(radem.shape) != 3 or len(xdata.shape) != 3:
        raise ValueError("chiArr should be a 1d array. radem and xdata should be 3d arrays.")
    if len(outputArray.shape) != 2 or len(sequence_lengths.shape) != 1:
        raise ValueError("outputArray should be a 2d array, sequence_lengths should be 1d.")

    if xdata.shape[0] == 0:
        raise ValueError("There must be at least one datapoint.")
    if xdata.shape[0] != outputArray.shape[0]:
        raise ValueError("The number of datapoints in the outputs and the inputs do "
                "not agree.")
    if radem.shape[0] != 3 or radem.shape[1] != 1:
        raise ValueError("radem must have length 3 for dim 0 and length 1 for dim1.")

    if outputArray.shape[1] % 2 != 0 or outputArray.shape[1] < 2:
        raise ValueError("Shape of output array is not appropriate.")
    
    if 2 * chiArr.shape[0] != outputArray.shape[1] or chiArr.shape[0] > radem.shape[2]:
        raise ValueError("Shape of output array and / or chiArr is inappropriate.")

    expectedNFreq = <double>(convWidth * xdata.shape[2])
    paddedBufferSize = <int>(2**math.ceil(np.log2(max(expectedNFreq, 2.)))  )
    if not radem.shape[2] % paddedBufferSize == 0:
        raise ValueError("radem should be an integer multiple of the padded width.")

    if sequence_lengths.shape[0] != xdata.shape[0]:
        raise ValueError("Inappropriate size for sequence_lengths.")
    if sequence_lengths.min() < convWidth or convWidth <= 0:
        raise ValueError("All elements of sequence_lengths must be >= convWidth.")
    if xdata.shape[1] < convWidth:
        raise ValueError("convWidth must be <= the shape of the input data.")
    if sequence_lengths.max() > xdata.shape[1]:
        raise ValueError("Maximum sequence length must be <= the shape of the input data.")
    
    if not radem.dtype == "int8":
        raise ValueError("radem must be int8.")

    if not outputArray.flags["C_CONTIGUOUS"] or not xdata.flags["C_CONTIGUOUS"] or not radem.flags["C_CONTIGUOUS"] \
        or not chiArr.flags["C_CONTIGUOUS"]:
        raise ValueError("One or more arguments is not C contiguous.")


    cdef uintptr_t addr_xdata = xdata.data.ptr
    cdef uintptr_t addr_radem = radem.data.ptr
    cdef uintptr_t addr_chi = chiArr.data.ptr
    cdef uintptr_t addr_seqlen = sequence_lengths.data.ptr
    cdef uintptr_t addr_output = outputArray.data.ptr

    scalingTerm = np.sqrt(1.0 / <double>chiArr.shape[0])

    if averageFeatures:
        scalingTerm /= (<double>xdata.shape[1] - <double>convWidth + 1.)

    if outputArray.dtype == "float64" and xdata.dtype == "float32" and \
            chiArr.dtype == "float32":
        errCode = convRBFFeatureGen[float](<int8_t*>addr_radem, <float*>addr_xdata,
                    <float*>addr_chi, <double*>addr_output, <int32_t*>addr_seqlen,
                    xdata.shape[0], xdata.shape[1],
                    xdata.shape[2], chiArr.shape[0], radem.shape[2],
                    convWidth, paddedBufferSize, scalingTerm)

    elif outputArray.dtype == "float64" and xdata.dtype == "float64" and \
            chiArr.dtype == "float64":
        errCode = convRBFFeatureGen[double](<int8_t*>addr_radem, <double*>addr_xdata,
                    <double*>addr_chi, <double*>addr_output, <int32_t*>addr_seqlen,
                    xdata.shape[0], xdata.shape[1],
                    xdata.shape[2], chiArr.shape[0], radem.shape[2],
                    convWidth, paddedBufferSize, scalingTerm)
    else:
        raise ValueError("Incorrect data types supplied.")

    if errCode.decode("UTF-8") != "no_error":
        raise Exception("Fatal error encountered while performing FHT RF generation.")



@cython.boundscheck(False)
@cython.wraparound(False)
def gpuConvGrad(xdata, sequence_lengths, radem, outputArray, chiArr,
                int convWidth, int numThreads, float sigma,
                bool averageFeatures = False):
    """Performs feature generation for the GraphRBF kernel while also performing
    gradient calculations.

    Args:
        xdata (cp.ndarray): An input 3d raw data array of shape (N x D x C)
            for N datapoints. D must be >= convWidth.
        sequence_lengths (cp.ndarray): A 1d array of shape[0]==N where each element
            is the length of the corresponding sequence in xdata. All values must be
            >= convWidth.
        radem (cp.ndarray): A stack of diagonal matrices with elements drawn from the
            Rademacher distribution. Shape must be (3 x D x C).
        outputArray (cp.ndarray): A numpy array in which the generated features will be
            stored. Is modified in-place.
        chiArr (cp.ndarray): A stack of diagonal matrices drawn from a chi distribution.
        convWidth (int): The width of the convolution. Must be <= D when xdata is (N x D x C).
        num_threads (int): Number of threads to use for FHT.
        sigma (float): The lengthscale.
        averageFeatures (bool): Whether to average the features generated along the
            first axis (makes kernel result less dependent on sequence length / graph
            size).

    Raises:
        ValueError: A ValueError is raised if unexpected or invalid inputs are supplied.

    Returns:
        gradient (cp.ndarray); An array of shape output.shape[0] x output.shape[1] x 1.
    """
    cdef const char *errCode
    cdef double scalingTerm
    cdef int num_repeats = (radem.shape[2] + xdata.shape[2] - 1) // xdata.shape[2]
    cdef int startPosition, cutoff, startPos2, cutoff2, i, j
    cdef double expectedNFreq
    cdef int paddedBufferSize

    if not isinstance(xdata, cp.ndarray) or not isinstance(sequence_lengths, cp.ndarray):
        raise ValueError("Passed non-cupy arrays to GPU function.")

    if len(chiArr.shape) != 1 or len(radem.shape) != 3 or len(xdata.shape) != 3:
        raise ValueError("chiArr should be a 1d array. radem and xdata should be 3d arrays.")
    if len(outputArray.shape) != 2:
        raise ValueError("outputArray should be a 2d array.")

    if xdata.shape[0] == 0:
        raise ValueError("There must be at least one datapoint.")
    if xdata.shape[0] != outputArray.shape[0]:
        raise ValueError("The number of datapoints in the outputs and the inputs do "
                "not agree.")
    if radem.shape[0] != 3 or radem.shape[1] != 1:
        raise ValueError("radem must have length 3 for dim 0 and length 1 for dim1.")

    if outputArray.shape[1] % 2 != 0 or outputArray.shape[1] < 2:
        raise ValueError("Shape of output array is not appropriate.")
    
    if 2 * chiArr.shape[0] != outputArray.shape[1] or chiArr.shape[0] > radem.shape[2]:
        raise ValueError("Shape of output array and / or chiArr is inappropriate.")

    expectedNFreq = <double>(convWidth * xdata.shape[2])
    paddedBufferSize = <int>(2**math.ceil(np.log2(max(expectedNFreq, 2.)))  )
    if not radem.shape[2] % paddedBufferSize == 0:
        raise ValueError("radem should be an integer multiple of the padded width.")

    if sequence_lengths.shape[0] != xdata.shape[0]:
        raise ValueError("Inappropriate size for sequence_lengths.")
    if sequence_lengths.min() < convWidth or convWidth <= 0:
        raise ValueError("All elements of sequence_lengths must be >= convWidth.")
    if xdata.shape[1] < convWidth:
        raise ValueError("convWidth must be <= the shape of the input data.")
    if sequence_lengths.max() > xdata.shape[1]:
        raise ValueError("Maximum sequence length must be <= the shape of the input data.")
    
    if not radem.dtype == "int8":
        raise ValueError("radem must be int8.")

    if not outputArray.flags["C_CONTIGUOUS"] or not xdata.flags["C_CONTIGUOUS"] or not radem.flags["C_CONTIGUOUS"] \
        or not chiArr.flags["C_CONTIGUOUS"]:
        raise ValueError("One or more arguments is not C contiguous.")


    gradient = cp.zeros((outputArray.shape[0], outputArray.shape[1], 1), dtype=cp.float64)
    if xdata.dtype == "float32":
        featureArray = cp.zeros((xdata.shape[0], xdata.shape[1], xdata.shape[2]),
                            dtype = cp.float32)
    else:
        featureArray = cp.zeros((xdata.shape[0], xdata.shape[1], xdata.shape[2]),
                            dtype = cp.float64)

    cdef uintptr_t addr_xdata = xdata.data.ptr
    cdef uintptr_t addr_radem = radem.data.ptr
    cdef uintptr_t addr_chi = chiArr.data.ptr
    cdef uintptr_t addr_seqlen = sequence_lengths.data.ptr
    cdef uintptr_t addr_output = outputArray.data.ptr
    cdef uintptr_t addr_gradient = gradient.data.ptr


    scalingTerm = np.sqrt(1.0 / <double>chiArr.shape[0])

    if averageFeatures:
        scalingTerm /= (<double>xdata.shape[1] - <double>convWidth + 1.)

    if outputArray.dtype == "float64" and xdata.dtype == "float32" and \
            chiArr.dtype == "float32":
        errCode = convRBFFeatureGrad[float](<int8_t*>addr_radem, <float*>addr_xdata,
            <float*>addr_chi, <double*>addr_output, <int32_t*>addr_seqlen,
            <double*>addr_gradient, sigma,
            xdata.shape[0], xdata.shape[1], xdata.shape[2],
            chiArr.shape[0], radem.shape[2], convWidth,
            paddedBufferSize, scalingTerm)

    elif outputArray.dtype == "float64" and xdata.dtype == "float64" and \
            chiArr.dtype == "float64":
        errCode = convRBFFeatureGrad[double](<int8_t*>addr_radem, <double*>addr_xdata,
            <double*>addr_chi, <double*>addr_output, <int32_t*>addr_seqlen,
            <double*>addr_gradient, sigma,
            xdata.shape[0], xdata.shape[1], xdata.shape[2],
            chiArr.shape[0], radem.shape[2], convWidth,
            paddedBufferSize, scalingTerm)
    else:
        raise ValueError("Incorrect data types supplied.")

    if errCode.decode("UTF-8") != "no_error":
        raise Exception("Fatal error encountered while performing FHT RF generation.")

    return gradient
