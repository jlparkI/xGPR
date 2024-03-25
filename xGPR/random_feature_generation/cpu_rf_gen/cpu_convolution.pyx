"""Wraps the C functions that generate random features for convolution
kernels & related kernels, for single-precision arithmetic.
Also performs all of the bounds and safety checks needed to use these
functions (the C functions do not do their own bounds checking). It
is EXTREMELY important that this wrapper not be bypassed for this
reason -- it double checks all of the array dimensions, types,
is data contiguous etc. before calling the wrapped C functions."""
import numpy as np
cimport numpy as np
cimport cython
from cython cimport floating
from libc cimport stdint
from libc.stdint cimport uintptr_t
from libc.stdint cimport int8_t, int32_t
from libcpp cimport bool
import math



cdef extern from "convolution_ops/conv1d_operations.h" nogil:
    const char *conv1dMaxpoolFeatureGen_[T](const int8_t *radem, const T xdata[],
            const T chiArr[], double *outputArray, const int32_t *seqlengths,
            int xdim0, int xdim1, int xdim2, int numThreads, int numFreqs,
            int convWidth, int paddedBufferSize)

cdef extern from "convolution_ops/rbf_convolution.h" nogil:
    const char *convRBFFeatureGen_[T](int8_t *radem, T xdata[],
            T chiArr[], double *outputArray, int32_t *seqlengths,
            int numThreads, int dim0,
            int dim1, int dim2,
            int numFreqs, int rademShape2,
            int convWidth, int paddedBufferSize,
            double scalingTerm, int scalingType)
    const char *convRBFGrad_[T](int8_t *radem, T xdata[],
            T chiArr[], double *outputArray, int32_t *seqlengths,
            double *gradientArray, T sigma,
            int numThreads, int dim0,
            int dim1, int dim2, int numFreqs,
            int rademShape2, int convWidth,
            int paddedBufferSize,
            double scalingTerm, int scalingType)


@cython.boundscheck(False)
@cython.wraparound(False)
def cpuConv1dMaxpool(np.ndarray[floating, ndim=3] xdata,
                np.ndarray[np.int32_t, ndim=1] sequence_lengths,
                np.ndarray[np.int8_t, ndim=3] radem,
                np.ndarray[np.float64_t, ndim=2] outputArray,
                np.ndarray[floating, ndim=1] chiArr,
                int convWidth,
                int numThreads):
    """Uses wrapped C extensions to perform random feature generation
    with ReLU activation and maxpooling.

    Args:
        xdata (np.ndarray): An input 3d raw data array of shape (N x D x C)
            for N datapoints. D must be >= convWidth. C must be a power of 2.
        sequence_lengths (np.ndarray): A 1d array of shape[0]==N where each element
            is the length of the corresponding sequence in xdata. All values must be
            >= convWidth.
        radem (np.ndarray): A stack of diagonal matrices with elements drawn from the
            Rademacher distribution. Shape must be (3 x D x C).
        outputArray (np.ndarray): A numpy array in which the generated features will be
            stored. Is modified in-place.
        chiArr (np.ndarray): A stack of diagonal matrices stored as an
            array of shape m * C drawn from a chi distribution.
        convWidth (int): The width of the convolution. Must be <= D when xdata is (N x D x C).
        num_threads (int): Number of threads to use for FHT.

    Raises:
        ValueError: A ValueError is raised if unexpected or invalid inputs are supplied.
    """
    cdef const char *errCode
    cdef double expectedNFreq
    cdef int paddedBufferSize
    cdef uintptr_t addr_input = xdata.ctypes.data
    cdef uintptr_t addr_chi = chiArr.ctypes.data
    
    if xdata.shape[0] == 0 or xdata.shape[0] != outputArray.shape[0]:
        raise ValueError("Incorrect input / output shapes.")


    if radem.shape[0] != 3 or radem.shape[1] != 1:
        raise ValueError("radem must have length 3 for dim 0 and length 1 for dim1.")
    if outputArray.shape[1] != radem.shape[2]:
        raise ValueError("outputArray.shape[1] must be an integer multiple of the next largest "
                "power of 2 greater than the kernel width * X.shape[2].")

    if chiArr.shape[0] != radem.shape[2]:
        raise ValueError("chiArr.shape[0] must == radem.shape[2].")
        
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

    if not outputArray.flags["C_CONTIGUOUS"] or not xdata.flags["C_CONTIGUOUS"] or not radem.flags["C_CONTIGUOUS"] \
        or not chiArr.flags["C_CONTIGUOUS"] or not sequence_lengths.flags["C_CONTIGUOUS"]:
        raise ValueError("One or more arguments is not C contiguous.")

    if chiArr.dtype == "float32" and xdata.dtype == "float32":
        errCode = conv1dMaxpoolFeatureGen_[float](&radem[0,0,0], <float*>addr_input,
                <float*>addr_chi, &outputArray[0,0], &sequence_lengths[0],
                xdata.shape[0], xdata.shape[1], xdata.shape[2], numThreads,
                radem.shape[2], convWidth, paddedBufferSize)

    elif chiArr.dtype == "float64" and xdata.dtype == "float64":
        errCode = conv1dMaxpoolFeatureGen_[double](&radem[0,0,0], <double*>addr_input,
                <double*>addr_chi, &outputArray[0,0], &sequence_lengths[0],
                xdata.shape[0], xdata.shape[1], xdata.shape[2], numThreads,
                radem.shape[2], convWidth, paddedBufferSize)

    else:
        raise ValueError("Unexpected types passed to wrapped C++ function.")

    if errCode.decode("UTF-8") != "no_error":
        raise Exception(errCode.decode("UTF-8"))



@cython.boundscheck(False)
@cython.wraparound(False)
def cpuConv1dFGen(np.ndarray[floating, ndim=3] xdata,
                np.ndarray[np.int32_t, ndim=1] sequence_lengths,
                np.ndarray[np.int8_t, ndim=3] radem,
                np.ndarray[np.float64_t, ndim=2] outputArray,
                np.ndarray[floating, ndim=1] chiArr,
                int convWidth, int numThreads,
                str averageFeatures = 'none'):
    """Uses wrapped C functions to generate random features for Conv1d RBF-related kernels.
    This function cannot be used to calculate the gradient so is only used for forward pass
    only (during fitting, inference, non-gradient-based optimization). It does not multiply
    by the lengthscales, so caller should do this.

    Args:
        xdata (np.ndarray): An input 3d raw data array of shape (N x D x C)
            for N datapoints. D must be >= convWidth. C must be a power of 2.
        sequence_lengths (np.ndarray): A 1d array of shape[0]==N where each element
            is the length of the corresponding sequence in xdata. All values must be
            >= convWidth.
        radem (np.ndarray): A stack of diagonal matrices with elements drawn from the
            Rademacher distribution. Shape must be (3 x D x C).
        outputArray (np.ndarray): A numpy array in which the generated features will be
            stored. Is modified in-place.
        chiArr (np.ndarray): A stack of diagonal matrices stored as an
            array of shape m * C drawn from a chi distribution.
        convWidth (int): The width of the convolution. Must be <= D when xdata is (N x D x C).
        num_threads (int): Number of threads to use for FHT.
        averageFeatures (str): Whether to average the features generated along the
            first axis (makes kernel result less dependent on sequence length / graph
            size). Must be one of 'none', 'sqrt', 'full'.

    Raises:
        ValueError: A ValueError is raised if unexpected or invalid inputs are supplied.
    """
    cdef const char *errCode
    cdef double scalingTerm
    cdef int scalingType
    cdef double expectedNFreq
    cdef int paddedBufferSize

    cdef uintptr_t addr_input = xdata.ctypes.data
    cdef uintptr_t addr_chi = chiArr.ctypes.data

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

    if not outputArray.flags["C_CONTIGUOUS"] or not xdata.flags["C_CONTIGUOUS"] or not radem.flags["C_CONTIGUOUS"] \
        or not chiArr.flags["C_CONTIGUOUS"] or not sequence_lengths.flags["C_CONTIGUOUS"]:
        raise ValueError("One or more arguments is not C contiguous.")

    scalingTerm = np.sqrt(1.0 / <double>(chiArr.shape[0]))
    scalingType = 0

    if averageFeatures == 'full':
        scalingType = 2
    elif averageFeatures == 'sqrt':
        scalingType = 1

    if chiArr.dtype == "float32" and xdata.dtype == "float32":
        errCode = convRBFFeatureGen_[float](&radem[0,0,0], <float*>addr_input,
                <float*>addr_chi, &outputArray[0,0], &sequence_lengths[0],
                numThreads, xdata.shape[0],
                xdata.shape[1], xdata.shape[2],
                chiArr.shape[0], radem.shape[2],
                convWidth, paddedBufferSize,
                scalingTerm, scalingType)

    elif chiArr.dtype == "float64" and xdata.dtype == "float64":
        errCode = convRBFFeatureGen_[double](&radem[0,0,0], <double*>addr_input,
                <double*>addr_chi, &outputArray[0,0], &sequence_lengths[0],
                numThreads, xdata.shape[0],
                xdata.shape[1], xdata.shape[2],
                chiArr.shape[0], radem.shape[2],
                convWidth, paddedBufferSize,
                scalingTerm, scalingType)

    else:
        raise ValueError("Unexpected types passed to wrapped C++ function.")

    if errCode.decode("UTF-8") != "no_error":
        raise Exception(errCode.decode("UTF-8"))





@cython.boundscheck(False)
@cython.wraparound(False)
def cpuConvGrad(np.ndarray[floating, ndim=3] xdata,
                np.ndarray[np.int32_t, ndim=1] sequence_lengths,
                np.ndarray[np.int8_t, ndim=3] radem,
                np.ndarray[np.float64_t, ndim=2] outputArray,
                np.ndarray[floating, ndim=1] chiArr,
                int convWidth, int numThreads, float sigma,
                str averageFeatures = 'none'):
    """Performs feature generation for RBF-based convolution kernels while
    also performing gradient calculations.

    Args:
        xdata (np.ndarray): An input 3d raw data array of shape (N x D x C)
            for N datapoints. D must be >= convWidth. C must be a power of 2.
        sequence_lengths (np.ndarray): A 1d array of shape[0]==N where each element
            is the length of the corresponding sequence in xdata. All values must be
            >= convWidth.
        radem (np.ndarray): A stack of diagonal matrices with elements drawn from the
            Rademacher distribution. Shape must be (3 x D x C).
        outputArray (np.ndarray): A numpy array in which the generated features will be
            stored. Is modified in-place.
        chiArr (np.ndarray): A stack of diagonal matrices stored as an
            array of shape m * C drawn from a chi distribution.
        convWidth (int): The width of the convolution. Must be <= D when xdata is (N x D x C).
        num_threads (int): Number of threads to use for FHT.
        sigma (float): The lengthscale.
        averageFeatures (str): Whether to average the features generated along the
            first axis (makes kernel result less dependent on sequence length / graph
            size). Must be one of 'none', 'sqrt', 'full'.

    Raises:
        ValueError: A ValueError is raised if unexpected or invalid inputs are supplied.

    Returns:
        gradient (np.ndarray); An array of shape output.shape[0] x output.shape[1] x 1.
    """
    cdef const char *errCode
    cdef np.ndarray[np.float64_t, ndim=3] gradient = np.zeros((outputArray.shape[0],
                            outputArray.shape[1], 1))
    cdef double scalingTerm
    cdef int scalingType

    cdef uintptr_t addr_input = xdata.ctypes.data
    cdef uintptr_t addr_chi = chiArr.ctypes.data

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

    if sequence_lengths.shape[0] != xdata.shape[0]:
        raise ValueError("Inappropriate size for sequence_lengths.")
    if sequence_lengths.min() < convWidth or convWidth <= 0:
        raise ValueError("All elements of sequence_lengths must be >= convWidth.")
    if xdata.shape[1] < convWidth:
        raise ValueError("convWidth must be <= the shape of the input data.")
    if sequence_lengths.max() > xdata.shape[1]:
        raise ValueError("Maximum sequence length must be <= the shape of the input data.")

    expectedNFreq = <double>(convWidth * xdata.shape[2])
    paddedBufferSize = <int>(2**math.ceil(np.log2(max(expectedNFreq, 2.)))  )
    if not radem.shape[2] % paddedBufferSize == 0:
        raise ValueError("radem should be an integer multiple of the padded width.")

    if not outputArray.flags["C_CONTIGUOUS"] or not xdata.flags["C_CONTIGUOUS"] or \
            not radem.flags["C_CONTIGUOUS"] or not chiArr.flags["C_CONTIGUOUS"]:
        raise ValueError("One or more arguments to a wrapped C++ func is not C contiguous.")

    scalingTerm = np.sqrt(1.0 / <double>(chiArr.shape[0]))
    scalingType = 0

    if averageFeatures == 'full':
        scalingType = 2
    elif averageFeatures == 'sqrt':
        scalingType = 1


    if chiArr.dtype == "float32" and xdata.dtype == "float32":
        errCode = convRBFGrad_[float](&radem[0,0,0], <float*>addr_input,
                    <float*>addr_chi, &outputArray[0,0], &sequence_lengths[0],
                    &gradient[0,0,0], sigma,
                    numThreads, xdata.shape[0],
                    xdata.shape[1], xdata.shape[2],
                    chiArr.shape[0], radem.shape[2],
                    convWidth, paddedBufferSize,
                    scalingTerm, scalingType)

    elif chiArr.dtype == "float64" and xdata.dtype == "float64":
        errCode = convRBFGrad_[double](&radem[0,0,0], <double*>addr_input,
                    <double*>addr_chi, &outputArray[0,0], &sequence_lengths[0],
                    &gradient[0,0,0], sigma,
                    numThreads, xdata.shape[0],
                    xdata.shape[1], xdata.shape[2],
                    chiArr.shape[0], radem.shape[2],
                    convWidth, paddedBufferSize,
                    scalingTerm, scalingType)

    else:
        raise ValueError("Unexpected types passed to wrapped C++ function.")

    if errCode.decode("UTF-8") != "no_error":
        raise Exception(errCode.decode("UTF-8"))

    return gradient
