"""Handles basic hadamard transform based operations, primarily
SORF and SRHT.

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


cdef extern from "basic_ops/basic_array_operations.h" nogil:
    const char *cudaSORF3d[T](T npArray[], np.int8_t *radem, 
                    int dim0, int dim1, int dim2)
    const char *cudaSRHT2d[T](T npArray[], 
                    int8_t *radem, int dim0,
                    int dim1)


@cython.boundscheck(False)
@cython.wraparound(False)
def cudaPySORFTransform(cpArray, radem, int numThreads):
    """This function performs the calculation of the structured
    orthogonal features or SORF approach to random Fourier features
    by wrapping floatCudaSORF3d, when the input array is an array
    of floats.
    Note that floatCudaSORF3d should ONLY be accessed through this wrapper
    since this wrapper performs key checks (the shape of the input
    arrays, are they C-contiguous etc.) that should not be bypassed.

    Args:
        cpArray (cp.ndarray): An array of type float32 on which the
            SORF operation will be performed in place. Must
            be of shape (N x D x C) where C is a power of 2.
        radem (cp.ndarray): A stack of diagonal matrices of type int8_t
            of shape (3 x D x C).
        num_threads (int): This argument is so that this function has
            the same interface as the CPU SORF Transform. It is not
            needed for the GPU transform and is ignored.
    """
    cdef const char *errCode
    cdef float logdim
    cdef uintptr_t addr = cpArray.data.ptr
    cdef uintptr_t addr_radem = radem.data.ptr

    #We need to know that cpArray.shape[1] and shape[2] match shape[1]
    #and shape[2] of radem, that all arrays are C contiguous and
    #have the correct data types, and that shape[2] of cpArray is a power
    #of two, which is a requirement for the transform.
    if cpArray.shape[0] == 0:
        raise ValueError("There must be at least one datapoint.")
    if cpArray.shape[1] != radem.shape[1] or cpArray.shape[2] != radem.shape[2]:
        raise ValueError("Incorrect array dims passed to floatCudaPySORFTransform.")
    if radem.shape[0] != 3:
        raise ValueError("radem must have length 3 for dim 0.")
    
    if not cpArray.flags["C_CONTIGUOUS"] or not radem.flags["C_CONTIGUOUS"]:
        raise ValueError("One or more arguments is not C contiguous.")
    if not radem.dtype == "int8":
        raise ValueError("radem must be of type int8.")
    logdim = np.log2(cpArray.shape[2])
    if np.ceil(logdim) != np.floor(logdim) or cpArray.shape[2] < 2:
        raise ValueError("dim2 of the input array to floatCudaPySORFTransform "
                            "must be a power of 2 >= 2.")


    if cpArray.dtype == "float32":
        errCode = cudaSORF3d[float](<float*>addr, <int8_t*>addr_radem,
                cpArray.shape[0], cpArray.shape[1],
                cpArray.shape[2])
    elif cpArray.dtype == "float64":
        errCode = cudaSORF3d[double](<double*>addr, <int8_t*>addr_radem,
                cpArray.shape[0], cpArray.shape[1],
                cpArray.shape[2])
    else:
        raise ValueError("Unexpected array type passed to a wrapped C++ function.")
    if errCode.decode("UTF-8") != "no_error":
        raise Exception("Fatal error encountered in floatCudaSORF3d.")





@cython.boundscheck(False)
@cython.wraparound(False)
def cudaSRHT(Z, radem,
                np.ndarray[np.int64_t, ndim=1] sampler,
                int compression_size,
                int numThreads):
    """Wraps the Cuda code for performing an SRHT operation.
    This wrapper performs all of the bounds checks,
    type checks etc and should not be bypassed.

    Args:
        Z (cp.ndarray): The array on which the transform will be performed.
            Transform is in place so nothing is returned. Shape is (N x C).
            C must be a power of 2.
        radem (cp.ndarray): A diagonal matrix with elements drawn from the
            Rademacher distribution. Shape must be (C).
        sampler (np.ndarray): An array containing indices that are used to permute
            the columns of Z post-transform. Shape must be < Z.shape[1].
        compression_size (int): The number of columns of Z that we plan
            to keep.
        numThreads (int): Not currently used, accepted only to preserve
            shared interface with CPU functions.
    """
    cdef const char *errCode
    cdef double dbl_scaling_factor;
    cdef uintptr_t addr = Z.data.ptr
    cdef uintptr_t addr_radem = radem.data.ptr

    if radem.dtype != "int8":
        raise ValueError("Incorrect data types passed.")
    if len(Z.shape) != 2 or len(radem.shape) != 1:
        raise ValueError("Incorrect array dims passed.")
    if Z.shape[0] == 0:
        raise ValueError("There must be at least one datapoint.")
    if Z.shape[1] != radem.shape[0] or sampler.shape[0] != compression_size:
        raise ValueError("Incorrect array dims passed.")
    if compression_size > Z.shape[1] or compression_size < 2:
        raise ValueError("Compression size must be <= num rffs but >= 2.")

    if not Z.flags["C_CONTIGUOUS"] or not radem.flags["C_CONTIGUOUS"]:
        raise ValueError("One or more arguments is not C contiguous.")
    logdim = np.log2(Z.shape[1])
    if np.ceil(logdim) != np.floor(logdim) or Z.shape[1] < 2:
        raise ValueError("dim1 of the input array must be a power of 2 >= 2.")

    if Z.dtype == "float32":
        errCode = cudaSRHT2d[float](<float*>addr, <int8_t*>addr_radem, Z.shape[0],
                Z.shape[1])
    elif Z.dtype == "float64":
        errCode = cudaSRHT2d[double](<double*>addr, <int8_t*>addr_radem, Z.shape[0],
                Z.shape[1])
    else:
        raise ValueError("Incorrect data types passed.")

    if errCode.decode("UTF-8") != "no_error":
        raise Exception("Fatal error encountered.")

    dbl_scaling_factor = np.sqrt( <double>radem.shape[0] / <double>compression_size )
    Z[:,:compression_size] = Z[:,sampler]
    Z[:,:compression_size] *= dbl_scaling_factor
