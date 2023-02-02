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
from libc cimport stdint
from libc.stdint cimport int8_t
import math

ctypedef stdint.int8_t cint8


cdef extern from "transform_functions.h" nogil:
    const char *fastHadamard3dFloatArray_(float *Z, int zDim0, int zDim1, int zDim2,
                        int numThreads)
    const char *fastHadamard3dDoubleArray_(double *Z, int zDim0, int zDim1, int zDim2,
                        int numThreads)
    
    const char *fastHadamard2dFloatArray_(float *Z, int zDim0, int zDim1,
                        int numThreads)
    const char *fastHadamard2dDoubleArray_(double *Z, int zDim0, int zDim1,
                        int numThreads)
    
    const char *SORFFloatBlockTransform_(float *Z, int8_t *radem, int zDim0,
        int zDim1, int zDim2, int numThreads)
    const char *SORFDoubleBlockTransform_(double *Z, int8_t *radem, int zDim0,
        int zDim1, int zDim2, int numThreads)

    const char *SRHTFloatBlockTransform_(float *Z, int8_t *radem,
            int zDim0, int zDim1, int numThreads)
    const char *SRHTDoubleBlockTransform_(double *Z, int8_t *radem,
            int zDim0, int zDim1, int numThreads)



@cython.boundscheck(False)
@cython.wraparound(False)
def doubleCpuFastHadamardTransform(np.ndarray[np.float64_t, ndim=3] Z,
                int numThreads):
    """Wraps fastHadamard3dDoubleArray_ from transform_functions.c and uses
    it to perform a fast Hadamard transform (unnormalized) on the last
    dimension of a 3d array of doubles. This function performs all of the bounds checks,
    type checks etc and should not be bypassed.

    Args:
        Z (np.ndarray): The array on which the transform will be performed.
            Transform is in place so nothing is returned. Shape is (N x D x C).
            C must be a  power of 2.
        numThreads (int): The number of threads to use.
    """
    cdef const char *errCode
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

    errCode = fastHadamard3dDoubleArray_(&Z[0,0,0], zDim0, zDim1, zDim2, numThreads)
    if errCode.decode("UTF-8") != "no_error":
        raise Exception("Fatal error encountered.")



@cython.boundscheck(False)
@cython.wraparound(False)
def floatCpuFastHadamardTransform(np.ndarray[np.float32_t, ndim=3] Z,
                int numThreads):
    """Wraps fastHadamard3dFloatArray_ from transform_functions.c and uses
    it to perform a fast Hadamard transform (unnormalized) on the last
    dimension of a 3d array of doubles. This function performs all of the bounds checks,
    type checks etc and should not be bypassed.

    Args:
        Z (np.ndarray): The array on which the transform will be performed.
            Transform is in place so nothing is returned. Shape is (N x D x C).
            C must be a  power of 2.
        numThreads (int): The number of threads to use.
    """
    cdef const char *errCode
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

    errCode = fastHadamard3dFloatArray_(&Z[0,0,0], zDim0, zDim1, zDim2, numThreads)
    if errCode.decode("UTF-8") != "no_error":
        raise Exception("Fatal error encountered.")



@cython.boundscheck(False)
@cython.wraparound(False)
def doubleCpuFastHadamardTransform2D(np.ndarray[np.float64_t, ndim=2] Z,
                int numThreads):
    """Wraps fastHadamard2dDoubleArray_ from transform_functions.c and uses
    it to perform a fast Hadamard transform (unnormalized) on the last
    dimension of a 2d array of doubles. This function performs all of the bounds checks,
    type checks etc and should not be bypassed.

    Args:
        Z (np.ndarray): The array on which the transform will be performed.
            Transform is in place so nothing is returned. Shape is (N x C).
            C must be a  power of 2.
        numThreads (int): The number of threads to use.
    """
    cdef const char *errCode
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

    errCode = fastHadamard2dDoubleArray_(&Z[0,0], zDim0, zDim1, numThreads)
    if errCode.decode("UTF-8") != "no_error":
        raise Exception("Fatal error encountered.")



@cython.boundscheck(False)
@cython.wraparound(False)
def floatCpuFastHadamardTransform2D(np.ndarray[np.float32_t, ndim=2] Z,
                int numThreads):
    """Wraps fastHadamard2dFloatArray_ from transform_functions.c and uses
    it to perform a fast Hadamard transform (unnormalized) on the last
    dimension of a 2d array of doubles. This function performs all of the bounds checks,
    type checks etc and should not be bypassed.

    Args:
        Z (np.ndarray): The array on which the transform will be performed.
            Transform is in place so nothing is returned. Shape is (N x C).
            C must be a  power of 2.
        numThreads (int): The number of threads to use.
    """
    cdef const char *errCode
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

    errCode = fastHadamard2dFloatArray_(&Z[0,0], zDim0, zDim1, numThreads)
    if errCode.decode("UTF-8") != "no_error":
        raise Exception("Fatal error encountered.")




@cython.boundscheck(False)
@cython.wraparound(False)
def doubleCpuSORFTransform(np.ndarray[np.float64_t, ndim=3] Z,
                np.ndarray[np.int8_t, ndim=3] radem,
                int numThreads):
    """Wraps SORFDoubleBlockTransform_ from transform_functions.c and uses
    it to perform the SORF operation on a 3d array of doubles.
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

    errCode = SORFDoubleBlockTransform_(&Z[0,0,0], &radem[0,0,0], zDim0, zDim1, zDim2,
                        numThreads)
    if errCode.decode("UTF-8") != "no_error":
        raise Exception("Fatal error encountered in cpuSORFTransform_.")




@cython.boundscheck(False)
@cython.wraparound(False)
def floatCpuSORFTransform(np.ndarray[np.float32_t, ndim=3] Z,
                np.ndarray[np.int8_t, ndim=3] radem,
                int numThreads):
    """Wraps SORFFloatBlockTransform_ from transform_functions.c and uses
    it to perform the SORF operation on a 3d array of floats.
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

    errCode = SORFFloatBlockTransform_(&Z[0,0,0], &radem[0,0,0], zDim0, zDim1, zDim2,
                        numThreads)
    if errCode.decode("UTF-8") != "no_error":
        raise Exception("Fatal error encountered in cpuSORFTransform_.")





@cython.boundscheck(False)
@cython.wraparound(False)
def doubleCpuSRHT(np.ndarray[np.float64_t, ndim=2] Z,
                np.ndarray[np.int8_t, ndim=1] radem,
                np.ndarray[np.int64_t, ndim=1] sampler,
                int compression_size,
                int numThreads):
    """Wraps SRHTDoubleBlockTransform_ from transform_functions.c and uses
    it to perform the SRHT operation on a 2d array of doubles.
    This wrapper performs all of the bounds checks,
    type checks etc and should not be bypassed.

    Args:
        Z (np.ndarray): The array on which the transform will be performed.
            Transform is in place so nothing is returned. Shape is (N x C).
            C must be a power of 2.
        radem (np.ndarray): A diagonal matrix with elements drawn from the
            Rademacher distribution. Shape must be (C).
        sampler (np.ndarray): An array containing indices that are used to permute
            the columns of Z post-transform. Shape must be < Z.shape[1].
        compression_size (int): The number of columns of Z that we plan
            to keep.
        numThreads (int): The number of threads to use.
    """
    cdef const char *errCode
    cdef double scaling_factor;

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

    errCode = SRHTDoubleBlockTransform_(&Z[0,0], &radem[0], zDim0, zDim1,
                        numThreads)
    if errCode.decode("UTF-8") != "no_error":
        raise Exception("Fatal error encountered.")

    Z[:,:compression_size] = Z[:,sampler]
    scaling_factor = np.sqrt( <double>radem.shape[0] / <double>compression_size )
    Z[:,:compression_size] *= scaling_factor



@cython.boundscheck(False)
@cython.wraparound(False)
def floatCpuSRHT(np.ndarray[np.float32_t, ndim=2] Z,
                np.ndarray[np.int8_t, ndim=1] radem,
                np.ndarray[np.int64_t, ndim=1] sampler,
                int compression_size,
                int numThreads):
    """Wraps SRHTFloatBlockTransform_ from transform_functions.c and uses
    it to perform the SRHT operation on a 2d array of floats.
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

    errCode = SRHTFloatBlockTransform_(&Z[0,0], &radem[0], zDim0, zDim1,
                        numThreads)
    if errCode.decode("UTF-8") != "no_error":
        raise Exception("Fatal error encountered.")

    Z[:,:compression_size] = Z[:,sampler]
    scaling_factor = np.sqrt( <float>radem.shape[0] / <float>compression_size )
    Z[:,:compression_size] *= scaling_factor
