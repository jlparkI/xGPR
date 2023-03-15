"""Handles random feature generation operations for poly &
graph poly kernels.

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
from libc.stdint cimport int8_t


cdef extern from "convolution_ops/convolution.h" nogil:
    const char *floatConv1dPrep(int8_t *radem,
                float *reshapedX, int reshapedDim0, 
                int reshapedDim1, int reshapedDim2,
                int startPosition, int numFreqs)
    const char *doubleConv1dPrep(int8_t *radem,
                double *reshapedX, int reshapedDim0, 
                int reshapedDim1, int reshapedDim2,
                int startPosition, int numFreqs)


cdef extern from "basic_ops/float_array_operations.h" nogil:
    const char *floatCudaSORF3d(float *npArray, np.int8_t *radem, 
                    int dim0, int dim1, int dim2)

cdef extern from "basic_ops/double_array_operations.h" nogil:
    const char *doubleCudaSORF3d(double *npArray, np.int8_t *radem, 
                    int dim0, int dim1, int dim2)



@cython.boundscheck(False)
@cython.wraparound(False)
def floatGpuGraphPolyFHT(reshapedX, radem, chiArr, outputArray, int polydegree,
                int numThreads):
    """Uses the wrapped PolyFHT_ and numpy operations to apply a pairwise
    polynomial kernel for graphs to float32 arrays.

    Args:
        reshapedX (cp.ndarray): Raw data reshaped so that the random features
            transformation can be applied. This array is not modified in place --
            rather the features that are generated are stored in outputArray. Shape is (N x D x C)
            for N datapoints. C must be a power of 2.
        radem (cp.ndarray): A stack of diagonal matrices with elements drawn from the
            Rademacher distribution. Shape must be (polydegree x 1 x C).
        chiArr (np.ndarray): A stack of diagonal matrices stored as an
            array of shape (polydegree, m * C) drawn from a chi distribution.
        outputArray (cp.ndarray): An array in which the generated features will be
            stored. Is modified in-place.
        polydegree (int): The degree of the polynomial kernel that we approximate. Should
            be <= 4 (for very high-degree polynomial kernels we are probably better off
            switching to an RBF or convolution kernel).
        num_threads (int): Number of threads to use for FHT. Not used for gpu,
            merely kept here for consistency with CPU version.

    Raises:
        ValueError: A ValueError is raised if unexpected or invalid inputs are supplied.
    """
    cdef const char *errCode
    reshapedXCopy = reshapedX.copy()
    preSumFeats = reshapedX.copy()
    cdef int num_repeats = (radem.shape[2] + reshapedX.shape[2] - 1) // reshapedX.shape[2]
    cdef int startPosition, cutoff, i, j

    if len(chiArr.shape) != 2 or len(radem.shape) != 3 or len(reshapedX.shape) != 3:
        raise ValueError("chiArr should be a 2d array. radem and reshapedX should be 3d arrays.")
    if len(outputArray.shape) != 2:
        raise ValueError("outputArray should be a 2d array.")

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
    
    if not radem.dtype == "int8":
        raise ValueError("radem must be int8.")
    if not outputArray.dtype == "float32" or not reshapedX.dtype == "float32":
        raise ValueError("reshapedX must be float32, outputArray must be float32.")
    if not chiArr.dtype == "float32":
        raise ValueError("chiArr must be float32.")

    cdef uintptr_t addr_reshapedCopy = reshapedXCopy.data.ptr
    cdef float *reshapedXCopyPtr = <float*>addr_reshapedCopy
    cdef uintptr_t addr_preSumFeats = preSumFeats.data.ptr
    cdef float *preSumFeatsPtr = <float*>addr_preSumFeats
 
    cdef uintptr_t addr_radem = radem.data.ptr
    cdef int8_t *radem_ptr = <int8_t*>addr_radem


    startPosition, cutoff = 0, reshapedX.shape[2]

    for i in range(num_repeats):
        preSumFeats[:] = reshapedX
        errCode = floatConv1dPrep(radem_ptr,
                    preSumFeatsPtr, reshapedX.shape[0], reshapedX.shape[1], 
                    reshapedX.shape[2], i * reshapedX.shape[2],
                    radem.shape[2])
        if errCode.decode("UTF-8") != "no_error":
            raise Exception("Fatal error encountered while performing FHT RF generation.")
        preSumFeats *= chiArr[0:1, None, startPosition:cutoff]

        for j in range(1, polydegree):
            reshapedXCopy[:] = reshapedX
            errCode = floatConv1dPrep(radem_ptr,
                    reshapedXCopyPtr, reshapedX.shape[0], reshapedX.shape[1], 
                    reshapedX.shape[2], i * reshapedX.shape[2] + (3 * j) * radem.shape[2],
                    radem.shape[2])
            reshapedXCopy *= chiArr[j:(j+1), None, startPosition:cutoff]
            preSumFeats *= reshapedXCopy
        outputArray[:,startPosition:cutoff] = cp.sum(preSumFeats, axis=1)

        cutoff += reshapedX.shape[2]
        startPosition += reshapedX.shape[2]





@cython.boundscheck(False)
@cython.wraparound(False)
def floatGpuPolyFHT(reshapedX, radem, chiArr, outputArray, int polydegree,
                int numThreads):
    """Polynomial kernel for fixed vector data to float32 arrays.

    Args:
        reshapedX (cp.ndarray): Raw data reshaped so that the random features
            transformation can be applied. This array is not modified in place --
            rather the features that are generated are stored in outputArray. Shape is (N x D x C)
            for N datapoints. C must be a power of 2.
        radem (cp.ndarray): A stack of diagonal matrices with elements drawn from the
            Rademacher distribution. Shape must be (polydegree x D x C).
        chiArr (cp.ndarray): A stack of diagonal matrices stored as an
            array of shape (polydegree, D, C) drawn from a chi distribution.
        outputArray (cp.ndarray): An array in which the generated features will be
            stored. Is modified in-place.
        polydegree (int): The degree of the polynomial kernel that we approximate. Should
            be <= 4 (for very high-degree polynomial kernels we are probably better off
            switching to an RBF or convolution kernel).
        num_threads (int): Number of threads to use for FHT. Not used for gpu,
            merely kept here for consistency with CPU version.

    Raises:
        ValueError: A ValueError is raised if unexpected or invalid inputs are supplied.
    """
    cdef const char *errCode
    reshapedXCopy = reshapedX.copy()
    cdef int j, k

    if len(chiArr.shape) != 3 or len(radem.shape) != 3 or len(reshapedX.shape) != 3:
        raise ValueError("chiArr, radem and reshapedX should be 3d arrays.")
    if len(outputArray.shape) != 3:
        raise ValueError("outputArray should be a 3d array.")

    if reshapedX.shape[0] == 0:
        raise ValueError("There must be at least one datapoint.")
    if reshapedX.shape[0] != outputArray.shape[0] or reshapedX.shape[1] != outputArray.shape[1] or\
            reshapedX.shape[2] != outputArray.shape[2]:
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
    
    if not radem.dtype == "int8":
        raise ValueError("radem must be int8.")
    if not outputArray.dtype == "float32" or not reshapedX.dtype == "float32":
        raise ValueError("reshapedX, outputArray must be float32.")
    if not chiArr.dtype == "float32":
        raise ValueError("chiArr must be int64.")

    cdef uintptr_t addr_reshapedCopy = reshapedXCopy.data.ptr
    cdef float *reshapedXCopyPtr = <float*>addr_reshapedCopy
    cdef uintptr_t addr_outputArray = outputArray.data.ptr
    cdef float *outputArrayPtr = <float*>addr_outputArray
 
    cdef uintptr_t addr_radem = radem.data.ptr
    cdef int8_t *radem_ptr = <int8_t*>addr_radem


    outputArray[:] = reshapedX
    errCode = floatCudaSORF3d(outputArrayPtr, radem_ptr, outputArray.shape[0],
                outputArray.shape[1], outputArray.shape[2])
    if errCode.decode("UTF-8") != "no_error":
        raise Exception("Fatal error encountered while performing graph convolution.")
    outputArray *= chiArr[0:1, :, :]

    for j in range(1, polydegree):
        reshapedXCopy[:] = reshapedX
        radem_ptr += 3 * radem.shape[2] * radem.shape[1]
        errCode = floatCudaSORF3d(reshapedXCopyPtr,
                radem_ptr, outputArray.shape[0], outputArray.shape[1],
                outputArray.shape[2])
        reshapedXCopy *= chiArr[j:j+1, :, :]
        outputArray *= reshapedXCopy







@cython.boundscheck(False)
@cython.wraparound(False)
def doubleGpuGraphPolyFHT(reshapedX, radem, chiArr, outputArray, int polydegree,
                int numThreads):
    """Polynomial kernel for graphs with float64 arrays.
    Args:
        reshapedX (cp.ndarray): Raw data reshaped so that the random features
            transformation can be applied. This array is not modified in place --
            rather the features that are generated are stored in outputArray. Shape is (N x D x C)
            for N datapoints. C must be a power of 2.
        radem (cp.ndarray): A stack of diagonal matrices with elements drawn from the
            Rademacher distribution. Shape must be (polydegree x 1 x C).
        chiArr (np.ndarray): A stack of diagonal matrices stored as an
            array of shape (polydegree, m * C) drawn from a chi distribution.
        outputArray (cp.ndarray): An array in which the generated features will be
            stored. Is modified in-place.
        polydegree (int): The degree of the polynomial kernel that we approximate. Should
            be <= 4 (for very high-degree polynomial kernels we are probably better off
            switching to an RBF or convolution kernel).
        num_threads (int): Number of threads to use for FHT. Not used for gpu,
            merely kept here for consistency with CPU version.
    Raises:
        ValueError: A ValueError is raised if unexpected or invalid inputs are supplied.
    """
    cdef const char *errCode
    reshapedXCopy = reshapedX.copy()
    preSumFeats = reshapedX.copy()
    cdef int num_repeats = (radem.shape[2] + reshapedX.shape[2] - 1) // reshapedX.shape[2]
    cdef int startPosition, cutoff, i, j

    if len(chiArr.shape) != 2 or len(radem.shape) != 3 or len(reshapedX.shape) != 3:
        raise ValueError("chiArr should be a 2d array. radem and reshapedX should be 3d arrays.")
    if len(outputArray.shape) != 2:
        raise ValueError("outputArray should be a 2d array.")

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
    
    if not radem.dtype == "int8":
        raise ValueError("radem must be int8.")
    if not outputArray.dtype == "float64" or not reshapedX.dtype == "float64":
        raise ValueError("reshapedX must be float64, outputArray must be float64.")
    if not chiArr.dtype == "float64":
        raise ValueError("chiArr must be float64.")

    cdef uintptr_t addr_reshapedCopy = reshapedXCopy.data.ptr
    cdef double *reshapedXCopyPtr = <double*>addr_reshapedCopy
    cdef uintptr_t addr_preSumFeats = preSumFeats.data.ptr
    cdef double *preSumFeatsPtr = <double*>addr_preSumFeats
 
    cdef uintptr_t addr_radem = radem.data.ptr
    cdef int8_t *radem_ptr = <int8_t*>addr_radem


    startPosition, cutoff = 0, reshapedX.shape[2]

    for i in range(num_repeats):
        preSumFeats[:] = reshapedX
        errCode = doubleConv1dPrep(radem_ptr,
                    preSumFeatsPtr, reshapedX.shape[0], reshapedX.shape[1], 
                    reshapedX.shape[2], i * reshapedX.shape[2],
                    radem.shape[2])
        if errCode.decode("UTF-8") != "no_error":
            raise Exception("Fatal error encountered while performing FHT RF generation.")
        preSumFeats *= chiArr[0:1, None, startPosition:cutoff]

        for j in range(1, polydegree):
            reshapedXCopy[:] = reshapedX
            errCode = doubleConv1dPrep(radem_ptr,
                    reshapedXCopyPtr, reshapedX.shape[0], reshapedX.shape[1], 
                    reshapedX.shape[2], i * reshapedX.shape[2] + (3 * j) * radem.shape[2],
                    radem.shape[2])
            reshapedXCopy *= chiArr[j:(j+1), None, startPosition:cutoff]
            preSumFeats *= reshapedXCopy
        outputArray[:,startPosition:cutoff] = cp.sum(preSumFeats, axis=1)

        cutoff += reshapedX.shape[2]
        startPosition += reshapedX.shape[2]





@cython.boundscheck(False)
@cython.wraparound(False)
def doubleGpuPolyFHT(reshapedX, radem, chiArr, outputArray, int polydegree,
                int numThreads):
    """Polynomial kernel for fixed vector data on float64 arrays.
    Args:
        reshapedX (cp.ndarray): Raw data reshaped so that the random features
            transformation can be applied. This array is not modified in place --
            rather the features that are generated are stored in outputArray. Shape is (N x D x C)
            for N datapoints. C must be a power of 2.
        radem (cp.ndarray): A stack of diagonal matrices with elements drawn from the
            Rademacher distribution. Shape must be (polydegree x D x m * C).
        chiArr (cp.ndarray): A stack of diagonal matrices stored as an
            array of shape (polydegree, D, C) drawn from a chi distribution.
        outputArray (cp.ndarray): An array in which the generated features will be
            stored. Is modified in-place.
        polydegree (int): The degree of the polynomial kernel that we approximate. Should
            be <= 4 (for very high-degree polynomial kernels we are probably better off
            switching to an RBF or convolution kernel).
        num_threads (int): Number of threads to use for FHT. Not used for gpu,
            merely kept here for consistency with CPU version.
    Raises:
        ValueError: A ValueError is raised if unexpected or invalid inputs are supplied.
    """
    cdef const char *errCode
    reshapedXCopy = reshapedX.copy()
    cdef int j

    if len(chiArr.shape) != 3 or len(radem.shape) != 3 or len(reshapedX.shape) != 3:
        raise ValueError("chiArr, radem and reshapedX should be 3d arrays.")
    if len(outputArray.shape) != 3:
        raise ValueError("outputArray should be a 3d array.")

    if reshapedX.shape[0] == 0:
        raise ValueError("There must be at least one datapoint.")
    if reshapedX.shape[0] != outputArray.shape[0] or reshapedX.shape[1] != outputArray.shape[1] or\
            reshapedX.shape[2] != outputArray.shape[2]:
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
    
    if not radem.dtype == "int8":
        raise ValueError("radem must be int8.")
    if not outputArray.dtype == "float64" or not reshapedX.dtype == "float64":
        raise ValueError("reshapedX, outputArray must be float64.")
    if not chiArr.dtype == "float64":
        raise ValueError("chiArr must be float64.")

    cdef uintptr_t addr_reshapedCopy = reshapedXCopy.data.ptr
    cdef double *reshapedXCopyPtr = <double*>addr_reshapedCopy
    cdef uintptr_t addr_outputArray = outputArray.data.ptr
    cdef double *outputArrayPtr = <double*>addr_outputArray
 
    cdef uintptr_t addr_radem = radem.data.ptr
    cdef int8_t *radem_ptr = <int8_t*>addr_radem



    outputArray[:] = reshapedX
    errCode = doubleCudaSORF3d(outputArrayPtr, radem_ptr, outputArray.shape[0],
                outputArray.shape[1], outputArray.shape[2])
    if errCode.decode("UTF-8") != "no_error":
        raise Exception("Fatal error encountered while performing graph convolution.")
    outputArray *= chiArr[0:1, :, :]

    for j in range(1, polydegree):
        reshapedXCopy[:] = reshapedX
        radem_ptr += 3 * radem.shape[2] * radem.shape[1]
        errCode = doubleCudaSORF3d(reshapedXCopyPtr,
                radem_ptr, outputArray.shape[0], outputArray.shape[1],
                outputArray.shape[2])
        reshapedXCopy *= chiArr[j:j+1, :, :]
        outputArray *= reshapedXCopy
