"""Wraps the C functions that perform the fastHadamardTransform on CPU,
the structured orthogonal features or SORF operations on CPU and the
fast Hadamard transform based SRHT on CPU.

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
from libc.stdint cimport int8_t
import math


cdef extern from "basic_ops/transform_functions.h" nogil:
    const char *fastHadamard3dArray_[T](T Z[], int zDim0, int zDim1, int zDim2,
                        int numThreads)
    const char *fastHadamard2dArray_[T](T Z[], int zDim0, int zDim1,
                        int numThreads)
    const char *SORFBlockTransform_[T](T Z[], int8_t *radem, int zDim0,
        int zDim1, int zDim2, int numThreads)
    const char *SRHTBlockTransform_[T](T Z[], int8_t *radem,
            int zDim0, int zDim1, int numThreads)





@cython.boundscheck(False)
@cython.wraparound(False)
def cpuFastHadamardTransform(np.ndarray[floating, ndim=3] Z,
                int numThreads):
    """Wraps fastHadamard3dArray_ from transform_functions.c and uses
    it to perform a fast Hadamard transform (unnormalized) on the last
    dimension of a 3d array. This function performs all of the bounds checks,
    type checks etc and should not be bypassed.

    Args:
        Z (np.ndarray): The array on which the transform will be performed.
            Transform is in place so nothing is returned. Shape is (N x D x C).
            C must be a  power of 2.
        numThreads (int): The number of threads to use.
    """
    cdef const char *errCode
    cdef uintptr_t addr_input = Z.ctypes.data

    if Z.shape[0] == 0:
        raise ValueError("There must be at least one datapoint.")
    if not Z.flags["C_CONTIGUOUS"]:
        raise ValueError("One or more arguments is not C contiguous.")
    if Z.shape[2] == 1:
        raise ValueError("dim2 of the input array must be > 1.")
    logdim = np.log2(Z.shape[2])
    if np.ceil(logdim) != np.floor(logdim) or Z.shape[2] < 2:
        raise ValueError("dim2 of the input array must be a power of 2 >= 2.")
    cdef int zDim0 = Z.shape[0]
    cdef int zDim1 = Z.shape[1]
    cdef int zDim2 = Z.shape[2]

    if Z.dtype == "float32":
        errCode = fastHadamard3dArray_[float](<float*>addr_input, zDim0, zDim1, zDim2, numThreads)
    elif Z.dtype == "float64":
        errCode = fastHadamard3dArray_[double](<double*>addr_input, zDim0, zDim1, zDim2, numThreads)
    else:
        raise ValueError("Unexpected types passed to wrapped C++ function.")

    if errCode.decode("UTF-8") != "no_error":
        raise Exception("Fatal error encountered.")






@cython.boundscheck(False)
@cython.wraparound(False)
def cpuFastHadamardTransform2D(np.ndarray[floating, ndim=2] Z,
                int numThreads):
    """Wraps fastHadamard2dArray_ from transform_functions.c and uses
    it to perform a fast Hadamard transform (unnormalized) on the last
    dimension of a 2d array. This function performs all of the bounds checks,
    type checks etc and should not be bypassed.

    Args:
        Z (np.ndarray): The array on which the transform will be performed.
            Transform is in place so nothing is returned. Shape is (N x C).
            C must be a  power of 2.
        numThreads (int): The number of threads to use.
    """
    cdef const char *errCode
    cdef uintptr_t addr_input = Z.ctypes.data

    if Z.shape[0] == 0:
        raise ValueError("There must be at least one datapoint.")
    if not Z.flags["C_CONTIGUOUS"]:
        raise ValueError("One or more arguments is not C contiguous.")
    if Z.shape[1] == 1:
        raise ValueError("dim1 of the input array must be > 1.")
    logdim = np.log2(Z.shape[1])
    if np.ceil(logdim) != np.floor(logdim) or Z.shape[1] < 2:
        raise ValueError("dim1 of the input array must be a power of 2 >= 2.")
    cdef int zDim0 = Z.shape[0]
    cdef int zDim1 = Z.shape[1]

    if Z.dtype == "float32":
        errCode = fastHadamard2dArray_[float](<float*>addr_input, zDim0, zDim1, numThreads)
    elif Z.dtype == "float64":
        errCode = fastHadamard2dArray_[double](<double*>addr_input, zDim0, zDim1, numThreads)
    else:
        raise ValueError("Unexpected types passed to wrapped C++ function.")

    if errCode.decode("UTF-8") != "no_error":
        raise Exception("Fatal error encountered.")







@cython.boundscheck(False)
@cython.wraparound(False)
def cpuSORFTransform(np.ndarray[floating, ndim=3] Z,
                np.ndarray[np.int8_t, ndim=3] radem,
                int numThreads):
    """Wraps SORFBlockTransform_ from transform_functions.c and uses
    it to perform the SORF operation on a 3d array.
    This wrapper performs all of the bounds checks,
    type checks etc and should not be bypassed.

    Args:
        Z (np.ndarray): The array on which the transform will be performed.
            Transform is in place so nothing is returned. Shape is (N x D x C).
            C must be a power of 2.
        radem (np.ndarray): A stack of diagonal matrices with elements drawn from the
            Rademacher distribution. Shape must be (3 x D x C).
        numThreads (int): The number of threads to use.
    """
    cdef const char *errCode
    cdef uintptr_t addr_input = Z.ctypes.data

    if Z.shape[0] == 0:
        raise ValueError("There must be at least one datapoint.")
    if Z.shape[1] != radem.shape[1] or Z.shape[2] != radem.shape[2]:
        raise ValueError("Incorrect array dims passed to cpuSORFTransform.")
    if radem.shape[0] != 3:
        raise ValueError("radem must have length 3 for dim 0.")
    
    if not Z.flags["C_CONTIGUOUS"] or not radem.flags["C_CONTIGUOUS"]:
        raise ValueError("One or more arguments to cpuSORFTransform is not "
                "C contiguous.")
    logdim = np.log2(Z.shape[2])
    if np.ceil(logdim) != np.floor(logdim) or Z.shape[2] < 2:
        raise ValueError("dim2 of the input array to cpuSORFTransform "
                            "must be a power of 2 >= 2.")
    cdef int zDim0 = Z.shape[0]
    cdef int zDim1 = Z.shape[1]
    cdef int zDim2 = Z.shape[2]

    if Z.dtype == "float32":
        errCode = SORFBlockTransform_[float](<float*>addr_input, &radem[0,0,0], zDim0, zDim1, zDim2,
                        numThreads)
    elif Z.dtype == "float64":
        errCode = SORFBlockTransform_[double](<double*>addr_input, &radem[0,0,0], zDim0, zDim1, zDim2,
                        numThreads)
    else:
        raise ValueError("Unexpected types passed to wrapped C++ function.")

    if errCode.decode("UTF-8") != "no_error":
        raise Exception("Fatal error encountered in cpuSORFTransform_.")







@cython.boundscheck(False)
@cython.wraparound(False)
def cpuSRHT(np.ndarray[floating, ndim=2] Z,
                np.ndarray[np.int8_t, ndim=1] radem,
                np.ndarray[np.int64_t, ndim=1] sampler,
                int compression_size,
                int numThreads):
    """Wraps SRHTBlockTransform_ from transform_functions.c.
    This wrapper performs all of the bounds checks,
    type checks etc and should not be bypassed.

    Args:
        Z (np.ndarray): The array on which the transform will be performed.
            Transform is in place so nothing is returned. Shape is (N x C).
            C must be a power of 2.
        radem (np.ndarray): A stack of diagonal matrices with elements drawn from the
            Rademacher distribution. Shape must be (C).
        sampler (np.ndarray): An array containing indices that are used to permute
            the columns of Z post-transform. Shape must be < Z.shape[1].
        compression_size (int): The number of columns of Z that we plan
            to keep.
        numThreads (int): The number of threads to use.
    """
    cdef const char *errCode
    cdef float scaling_factor
    cdef uintptr_t addr_input = Z.ctypes.data

    if Z.shape[0] == 0:
        raise ValueError("There must be at least one datapoint.")
    if Z.shape[1] != radem.shape[0]:
        raise ValueError("Incorrect array dims passed.")
    if sampler.shape[0] != compression_size:
        raise ValueError("Incorrect array dims passed.")
    if compression_size > Z.shape[1] or compression_size < 2:
        raise ValueError("Compression size must be <= num rffs but >= 2.")
    
    if not Z.flags["C_CONTIGUOUS"] or not radem.flags["C_CONTIGUOUS"]:
        raise ValueError("One or more arguments to cpuSORFTransform is not "
                "C contiguous.")
    logdim = np.log2(Z.shape[1])
    if np.ceil(logdim) != np.floor(logdim) or Z.shape[1] < 2:
        raise ValueError("dim1 of the input array must be a power of 2 >= 2.")
    cdef int zDim0 = Z.shape[0]
    cdef int zDim1 = Z.shape[1]

    if Z.dtype == "float32":
        errCode = SRHTBlockTransform_[float](<float*>addr_input, &radem[0], zDim0, zDim1,
                        numThreads)
    elif Z.dtype == "float64":
        errCode = SRHTBlockTransform_[double](<double*>addr_input, &radem[0], zDim0, zDim1,
                        numThreads)
    else:
        raise ValueError("Unexpected types passed to wrapped C++ function.")

    if errCode.decode("UTF-8") != "no_error":
        raise Exception("Fatal error encountered.")

    Z[:,:compression_size] = Z[:,sampler]
    scaling_factor = np.sqrt( <float>radem.shape[0] / <float>compression_size )
    Z[:,:compression_size] *= scaling_factor
