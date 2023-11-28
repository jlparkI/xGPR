"""Wraps the C functions that generate random features for poly &
graph poly kernels.

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
from libc.stdint cimport int8_t, int32_t
import math



cdef extern from "convolution_ops/conv1d_operations.h" nogil:
    const char *conv1dPrep_[T](int8_t *radem, T reshapedX[],
            int numThreads, int reshapedDim0,
            int reshapedDim1, int reshapedDim2,
            int startPosition, int numFreqs)


cdef extern from "poly_ops/polynomial_operations.h" nogil:
    const char *cpuExactQuadratic_[T](T inArray[], double *outArray,
                int inDim0, int inDim1, int numThreads)
    const char *approxPolynomial_[T](int8_t *radem, T reshapedX[],
            T copyBuffer[], T chiArr[], double *outputArray,
            int numThreads, int polydegree, int reshapedDim0,
            int reshapedDim1, int reshapedDim2,
            int numFreqs, int rademShape2);


@cython.boundscheck(False)
@cython.wraparound(False)
def cpuGraphPolyFHT(np.ndarray[floating, ndim=3] reshapedX,
                np.ndarray[np.int8_t, ndim=3] radem,
                np.ndarray[floating, ndim=2] chiArr,
                np.ndarray[floating, ndim=2] outputArray,
                int polydegree, int numThreads):
    """Uses the wrapped PolyFHT_ and numpy operations to apply a pairwise
    polynomial kernel to all elements of two graphs.

    Args:
        reshapedX (np.ndarray): Raw data reshaped so that the random features
            transformation can be applied. This array is not modified in place --
            rather the features that are generated are stored in outputArray. Shape is (N x D x C)
            for N datapoints. C must be a power of 2.
        radem (np.ndarray): A stack of diagonal matrices with elements drawn from the
            Rademacher distribution. Shape must be (polydegree x 1 x C).
        chiArr (np.ndarray): A stack of diagonal matrices stored as an
            array of shape (polydegree, m * C) drawn from a chi distribution.
        outputArray (np.ndarray): A numpy array in which the generated features will be
            stored. Is modified in-place.
        polydegree (int): The degree of the polynomial kernel that we approximate. Should
            be <= 4 (for very high-degree polynomial kernels we are probably better off
            switching to an RBF or convolution kernel).
        num_threads (int): Number of threads to use for FHT.

    Raises:
        ValueError: A ValueError is raised if unexpected or invalid inputs are supplied.
    """
    cdef const char *errCode
    cdef np.ndarray[floating, ndim=3] reshapedXCopy = reshapedX.copy()
    cdef np.ndarray[floating, ndim=3] preSumFeats = reshapedX.copy()
    cdef int num_repeats = (radem.shape[2] + reshapedX.shape[2] - 1) // reshapedX.shape[2]
    cdef int startPosition, cutoff, i, j
    
    cdef uintptr_t addr_xcopy = reshapedXCopy.ctypes.data
    cdef uintptr_t addr_pre_sum_feats = preSumFeats.ctypes.data
    

    if reshapedX.shape[0] == 0:
        raise ValueError("There must be at least one datapoint.")
    if reshapedX.shape[0] != outputArray.shape[0]:
        raise ValueError("The number of datapoints in the outputs and the inputs do "
                "not agree.")
    if radem.shape[0] != 3 * polydegree or radem.shape[1] != 1:
        raise ValueError("radem must have length polydegree for dim 0 and length 1 for dim1.")
    if outputArray.shape[1] != radem.shape[2]:
        raise ValueError("outputArray.shape[1] must be radem.shape[2], which must be an integer multiple of "
                    "the next power of 2 greater than the kernel width * X.shape[2].")
    
    if chiArr.shape[1] != radem.shape[2]:
        raise ValueError("chiArr.shape[1] must == radem.shape[2].")
    if chiArr.shape[0] != polydegree:
        raise ValueError("chiArr.shape[0] must == polydegree.")
    logdim = np.log2(reshapedX.shape[2])
    if np.ceil(logdim) != np.floor(logdim) or reshapedX.shape[2] < 2:
        raise ValueError("dim2 of the reshapedX array must be a power of 2 >= 2.")
    if not radem.shape[2] % reshapedX.shape[2] == 0:
        raise ValueError("The number of sampled frequencies should be an integer multiple of "
                "reshapedX.shape[2].")

    if not outputArray.flags["C_CONTIGUOUS"] or not reshapedX.flags["C_CONTIGUOUS"] or not radem.flags["C_CONTIGUOUS"] \
        or not chiArr.flags["C_CONTIGUOUS"]:
        raise ValueError("One or more arguments is not C contiguous.")

    startPosition, cutoff = 0, reshapedX.shape[2]

    if reshapedX.dtype == "float32" and chiArr.dtype == "float32" and outputArray.dtype == "float32":
        for i in range(num_repeats):
            preSumFeats[:] = reshapedX
            errCode = conv1dPrep_[float](&radem[0,0,0],
                    <float*>addr_pre_sum_feats, numThreads,
                    reshapedX.shape[0], reshapedX.shape[1], 
                    reshapedX.shape[2], i * reshapedX.shape[2],
                    radem.shape[2])

            if errCode.decode("UTF-8") != "no_error":
                raise Exception("Fatal error encountered while performing FHT RF generation.")
            preSumFeats *= chiArr[0:1, None, startPosition:cutoff]
            for j in range(1, polydegree):
                reshapedXCopy[:] = reshapedX
                errCode = conv1dPrep_[float](&radem[3*j,0,0],
                    <float*>addr_xcopy, numThreads,
                    reshapedX.shape[0], reshapedX.shape[1], 
                    reshapedX.shape[2], i * reshapedX.shape[2],
                    radem.shape[2])
                reshapedXCopy *= chiArr[j:(j+1), None, startPosition:cutoff]
                preSumFeats *= reshapedXCopy

            outputArray[:,startPosition:cutoff] = np.sum(preSumFeats, axis=1)

            cutoff += reshapedX.shape[2]
            startPosition += reshapedX.shape[2]
    elif reshapedX.dtype == "float64" and chiArr.dtype == "float64" and outputArray.dtype == "float64":
        for i in range(num_repeats):
            preSumFeats[:] = reshapedX
            errCode = conv1dPrep_[double](&radem[0,0,0],
                    <double*>addr_pre_sum_feats, numThreads,
                    reshapedX.shape[0], reshapedX.shape[1], 
                    reshapedX.shape[2], i * reshapedX.shape[2],
                    radem.shape[2])

            if errCode.decode("UTF-8") != "no_error":
                raise Exception("Fatal error encountered while performing FHT RF generation.")
            preSumFeats *= chiArr[0:1, None, startPosition:cutoff]
            for j in range(1, polydegree):
                reshapedXCopy[:] = reshapedX
                errCode = conv1dPrep_[double](&radem[3*j,0,0],
                    <double*>addr_xcopy, numThreads,
                    reshapedX.shape[0], reshapedX.shape[1], 
                    reshapedX.shape[2], i * reshapedX.shape[2],
                    radem.shape[2])
                reshapedXCopy *= chiArr[j:(j+1), None, startPosition:cutoff]
                preSumFeats *= reshapedXCopy

            outputArray[:,startPosition:cutoff] = np.sum(preSumFeats, axis=1)

            cutoff += reshapedX.shape[2]
            startPosition += reshapedX.shape[2]



@cython.boundscheck(False)
@cython.wraparound(False)
def cpuPolyFHT(np.ndarray[floating, ndim=3] reshapedX,
                np.ndarray[np.int8_t, ndim=3] radem,
                np.ndarray[floating, ndim=3] chiArr,
                np.ndarray[np.float64_t, ndim=2] outputArray,
                int polydegree, int numThreads, int numFreqs):
    """Uses the wrapped PolyFHT_ and numpy operations to apply a pairwise
    polynomial kernel to all elements of two graphs.

    Args:
        reshapedX (np.ndarray): Raw data reshaped so that the random features
            transformation can be applied. This array is not modified in place --
            rather the features that are generated are stored in outputArray. Shape is (N x D x C)
            for N datapoints. C must be a power of 2.
        radem (np.ndarray): A stack of diagonal matrices with elements drawn from the
            Rademacher distribution. Shape must be (3 * polydegree x D x C).
        chiArr (np.ndarray): A stack of diagonal matrices stored as an
            array of shape (polydegree, D, C) drawn from a chi distribution.
        outputArray (np.ndarray): A numpy array in which the generated features will be
            stored. Is modified in-place. Of shape (N, numFreqs).
        polydegree (int): The degree of the polynomial kernel that we approximate. Should
            be <= 4 (for very high-degree polynomial kernels we are probably better off
            switching to an RBF or convolution kernel).
        numThreads (int): Number of threads to use for FHT.
        numFreqs (int): The number of random features requested.

    Raises:
        ValueError: A ValueError is raised if unexpected or invalid inputs are supplied.
    """
    cdef const char *errCode
    cdef np.ndarray[floating, ndim=3] reshapedXCopy = reshapedX.copy()
    cdef uintptr_t addr_input = reshapedX.ctypes.data
    cdef uintptr_t addr_cbuffer = reshapedXCopy.ctypes.data
    cdef uintptr_t addr_chi = chiArr.ctypes.data
    

    if reshapedX.shape[0] == 0:
        raise ValueError("There must be at least one datapoint.")
    if reshapedX.shape[0] != outputArray.shape[0] or numFreqs != outputArray.shape[1] or\
            numFreqs > (reshapedX.shape[1] * reshapedX.shape[2]):
        raise ValueError("The number of datapoints in the outputs and the inputs do "
                "not agree.")
    if radem.shape[0] != 3 * polydegree or chiArr.shape[0] != polydegree:
        raise ValueError("radem & chiArr must have length polydegree for dim 0.")
    
    if chiArr.shape[2] != radem.shape[2] or chiArr.shape[1] != radem.shape[1]:
        raise ValueError("chiArr must have same shape[1] and shape[2] as radem.")
    logdim = np.log2(reshapedX.shape[2])
    if np.ceil(logdim) != np.floor(logdim) or reshapedX.shape[2] < 2:
        raise ValueError("dim2 of the reshapedX array must be a power of 2 >= 2.")
    if radem.shape[2] != reshapedX.shape[2] or radem.shape[1] != reshapedX.shape[1]:
        raise ValueError("reshapedX shape[1] and shape[2] must == radem shape[1] and shape[2].")

    if not outputArray.flags["C_CONTIGUOUS"] or not reshapedX.flags["C_CONTIGUOUS"] or not radem.flags["C_CONTIGUOUS"] \
        or not chiArr.flags["C_CONTIGUOUS"]:
        raise ValueError("One or more arguments is not C contiguous.")



    if reshapedX.dtype == "float32" and chiArr.dtype == "float32":
        errCode = approxPolynomial_[float](&radem[0,0,0], <float*>addr_input,
            <float*>addr_cbuffer, <float*>addr_chi, &outputArray[0,0],
            numThreads, polydegree, reshapedX.shape[0], reshapedX.shape[1],
            reshapedX.shape[2], numFreqs);

        if errCode.decode("UTF-8") != "no_error":
            raise Exception("Fatal error encountered while generating random features.")

    elif reshapedX.dtype == "float64" and chiArr.dtype == "float64":
        errCode = approxPolynomial_[double](&radem[0,0,0], <double*>addr_input,
            <double*>addr_cbuffer, <double*>addr_chi, &outputArray[0,0],
            numThreads, polydegree, reshapedX.shape[0], reshapedX.shape[1],
            reshapedX.shape[2], numFreqs);

        if errCode.decode("UTF-8") != "no_error":
            raise Exception("Fatal error encountered while generating random features.")

    else:
        raise ValueError("Unexpected types passed to wrapped C++ function.")



@cython.boundscheck(False)
@cython.wraparound(False)
def cpuExactQuadratic(np.ndarray[floating, ndim=2] inputArray,
                np.ndarray[double, ndim=2] outputArray,
                int numThreads):
    """Wraps C++ operations for generating features for an exact
    quadratic.

    Args:
        inputArray (np.ndarray): The input data. This is not modified.
        outputArray (np.ndarray): The output array. Must have the appropriate
            shape such that all of the quadratic polynomial features can
            be written to it. The last column is assumed to be saved for 1
            for a y-intercept term.
        num_threads (int): Number of threads.

    Raises:
        ValueError: A ValueError is raised if unexpected or invalid inputs are supplied.
    """
    cdef const char *errCode
    cdef uintptr_t addr_input = inputArray.ctypes.data
    cdef int numExpectedFeats = int( inputArray.shape[1] * (inputArray.shape[1] - 1) / 2)
    numExpectedFeats += 2 * inputArray.shape[1] + 1

    if inputArray.shape[0] == 0:
        raise ValueError("There must be at least one datapoint.")
    if inputArray.shape[0] != outputArray.shape[0]:
        raise ValueError("The number of datapoints in the outputs and the inputs do "
                "not agree.")
    if outputArray.shape[1] != numExpectedFeats:
        raise ValueError("The shape of the output array is incorrect for a quadratic.")

    if not outputArray.flags["C_CONTIGUOUS"] or not inputArray.flags["C_CONTIGUOUS"]:
        raise ValueError("One or more arguments is not C contiguous.")

    if inputArray.dtype == "float32":
        errCode = cpuExactQuadratic_[float](<float*>addr_input, &outputArray[0,0],
                        inputArray.shape[0], inputArray.shape[1], numThreads)

    elif inputArray.dtype == "float64":
        errCode = cpuExactQuadratic_[double](<double*>addr_input, &outputArray[0,0],
                        inputArray.shape[0], inputArray.shape[1], numThreads)

    else:
        raise ValueError("Unexpected types passed to wrapped C++ function.")

    if errCode.decode("UTF-8") != "no_error":
        raise Exception("Fatal error encountered while performing graph convolution.")
