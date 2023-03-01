"""Handles basic hadamard transform based operations, primarily
SORF, SRHT and RBF / Matern feature generation.

Also performs all of the bounds and safety checks needed to use these
functions (the C functions do not do their own bounds checking). It
is EXTREMELY important that this wrapper not be bypassed for this
reason -- it double checks all of the array dimensions, types,
is data contiguous etc. before calling the wrapped C functions."""
import os
import numpy as np
cimport numpy as np
cimport cython
from libc cimport stdint
import cupy as cp
from libc.stdint cimport uintptr_t
import math
from libc.stdint cimport int8_t


cdef extern from "float_array_operations.h" nogil:
    const char *floatCudaSORF3d(float *npArray, np.int8_t *radem, 
                    int dim0, int dim1, int dim2)
    const char *floatCudaSRHT2d(float *npArray, 
                    int8_t *radem, int dim0, int dim1)

cdef extern from "double_array_operations.h" nogil:
    const char *doubleCudaSORF3d(double *npArray, np.int8_t *radem, 
                    int dim0, int dim1, int dim2)
    const char *doubleCudaSRHT2d(double *npArray, 
                    int8_t *radem, int dim0, int dim1)


cdef extern from "rbf_ops/double_rbf_ops.h" nogil:
    const char *doubleRBFFeatureGen(double *cArray, int8_t *radem,
                double *chiArr, double *outputArray,
                double rbfNormConstant,
                int dim0, int dim1, int dim2,
                int numFreqs)

cdef extern from "rbf_ops/float_rbf_ops.h" nogil:
    const char *floatRBFFeatureGen(float *cArray, int8_t *radem,
                float *chiArr, double *outputArray,
                double rbfNormConstant,
                int dim0, int dim1, int dim2,
                int numFreqs)




@cython.boundscheck(False)
@cython.wraparound(False)
def floatCudaPySORFTransform(cpArray, radem, int numThreads):
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
        raise ValueError("One or more arguments to floatCudaPySORFTransform is not "
                "C contiguous.")
    if not cpArray.dtype == "float32":
        raise ValueError("The input cupy array to floatCudaPySORFTransform must be "
                "an array of type float32.")
    if not radem.dtype == "int8":
        raise ValueError("radem must be of type int8.")
    logdim = np.log2(cpArray.shape[2])
    if np.ceil(logdim) != np.floor(logdim) or cpArray.shape[2] < 2:
        raise ValueError("dim2 of the input array to floatCudaPySORFTransform "
                            "must be a power of 2 >= 2.")

    #Access the first element of each array. Note we assume
    #these arrays already live on GPU -- that should always be
    #true if using this function -- copying arrays back and forth 
    #more often than needed is very expensive and not recommended
    cdef uintptr_t addr = cpArray.data.ptr
    cdef float *cArray = <float*>addr
    
    cdef uintptr_t addr_radem = radem.data.ptr
    cdef int8_t *radem_ptr = <int8_t*>addr_radem

    errCode = floatCudaSORF3d(cArray, radem_ptr,
                cpArray.shape[0], cpArray.shape[1],
                cpArray.shape[2])
    if errCode.decode("UTF-8") != "no_error":
        raise Exception("Fatal error encountered in floatCudaSORF3d.")





@cython.boundscheck(False)
@cython.wraparound(False)
def doubleCudaPySORFTransform(cpArray, radem, int numThreads):
    """This function performs the calculation of the structured
    orthogonal features or SORF approach to random Fourier features
    by wrapping doubleCudaSORF3d, when the input array is an array of
    doubles.
    Note that doubleCudaSORF3d should ONLY be accessed through this wrapper
    since this wrapper performs key checks (the shape of the input
    arrays, are they C-contiguous etc.) that should not be bypassed.

    Args:
        cpArray (cp.ndarray): An array of type float64 on which the
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

    #We need to know that cpArray.shape[1] and shape[2] match shape[1]
    #and shape[2] of radem, that all arrays are C contiguous and
    #have the correct data types, and that shape[2] of cpArray is a power
    #of two, which is a requirement for the transform.
    if cpArray.shape[0] == 0:
        raise ValueError("There must be at least one datapoint.")
    if cpArray.shape[1] != radem.shape[1] or cpArray.shape[2] != radem.shape[2]:
        raise ValueError("Incorrect array dims passed to doubleCudaPySORFTransform.")
    if radem.shape[0] != 3:
        raise ValueError("radem must have length 3 for dim 0.")
    
    if not cpArray.flags["C_CONTIGUOUS"] or not radem.flags["C_CONTIGUOUS"]:
        raise ValueError("One or more arguments to doubleCudaPySORFTransform is not "
                "C contiguous.")
    if not cpArray.dtype == "float64":
        raise ValueError("The input cupy array to doubleCudaPySORFTransform must be "
                "an array of type float64.")
    if not radem.dtype == "int8":
        raise ValueError("radem must be of type int8.")
    logdim = np.log2(cpArray.shape[2])
    if np.ceil(logdim) != np.floor(logdim) or cpArray.shape[2] < 2:
        raise ValueError("dim2 of the input array to doubleCudaPySORFTransform "
                            "must be a power of 2 >= 2.")

    #Access the first element of each array. Note we assume
    #these arrays already live on GPU -- that should always be
    #true if using this function -- copying arrays back and forth 
    #more often than needed is very expensive and not recommended
    cdef uintptr_t addr = cpArray.data.ptr
    cdef double *cArray = <double*>addr
    
    cdef uintptr_t addr_radem = radem.data.ptr
    cdef int8_t *radem_ptr = <int8_t*>addr_radem

    errCode = doubleCudaSORF3d(cArray, radem_ptr,
                cpArray.shape[0], cpArray.shape[1],
                cpArray.shape[2])
    if errCode.decode("UTF-8") != "no_error":
        raise Exception("Fatal error encountered in doubleCudaSORF3d.")




@cython.boundscheck(False)
@cython.wraparound(False)
def doubleCudaSRHT(Z, radem,
                np.ndarray[np.int64_t, ndim=1] sampler,
                int compression_size,
                int numThreads):
    """Wraps SRHTDoubleBlockTransform from transform_functions.c and uses
    it to perform the SRHT operation on a 2d array of doubles.
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
    cdef double scaling_factor;

    if len(Z.shape) != 2 or len(radem.shape) != 1:
        raise ValueError("Incorrect array dims passed.")
    if not Z.dtype == "float64" or not radem.dtype == "int8":
        raise ValueError("Incorrect data types passed.")
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
    cdef uintptr_t addr = Z.data.ptr
    cdef double *ZArray = <double*>addr
    
    cdef uintptr_t addr_radem = radem.data.ptr
    cdef int8_t *radem_ptr = <int8_t*>addr_radem

    errCode = doubleCudaSRHT2d(ZArray, radem_ptr, zDim0, zDim1)
    if errCode.decode("UTF-8") != "no_error":
        raise Exception("Fatal error encountered.")

    Z[:,:compression_size] = Z[:,sampler]
    scaling_factor = np.sqrt( <double>radem.shape[0] / <double>compression_size )
    Z[:,:compression_size] *= scaling_factor



@cython.boundscheck(False)
@cython.wraparound(False)
def floatCudaSRHT(Z, radem,
                np.ndarray[np.int64_t, ndim=1] sampler,
                int compression_size,
                int numThreads):
    """Wraps SRHTFloatBlockTransform from transform_functions.c and uses
    it to perform the SRHT operation on a 2d array of floats.
    This wrapper performs all of the bounds checks,
    type checks etc and should not be bypassed.

    Args:
        Z (cp.ndarray): The array on which the transform will be performed.
            Transform is in place so nothing is returned. Shape is (N x C).
            C must be a power of 2.
        radem (cp.ndarray): A stack of diagonal matrices with elements drawn from the
            Rademacher distribution. Shape must be (C).
        sampler (np.ndarray): An array containing indices that are used to permute
            the columns of Z post-transform. Shape must be < Z.shape[1].
        compression_size (int): The number of columns of Z that we plan
            to keep.
        numThreads (int): Not used, accepted only to preserve shared interface
            with CPU functions.
    """
    cdef const char *errCode
    cdef float scaling_factor

    if len(Z.shape) != 2 or len(radem.shape) != 1:
        raise ValueError("Incorrect array dims passed.")
    if not Z.dtype == "float32" or not radem.dtype == "int8":
        raise ValueError("Incorrect data types passed.")
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
    cdef uintptr_t addr = Z.data.ptr
    cdef float *ZArray = <float*>addr

    cdef uintptr_t addr_radem = radem.data.ptr
    cdef int8_t *radem_ptr = <int8_t*>addr_radem

    errCode = floatCudaSRHT2d(ZArray, radem_ptr, zDim0, zDim1)
    if errCode.decode("UTF-8") != "no_error":
        raise Exception("Fatal error encountered.")

    Z[:,:compression_size] = Z[:,sampler]
    scaling_factor = np.sqrt( <float>radem.shape[0] / <float>compression_size )
    Z[:,:compression_size] *= scaling_factor



@cython.boundscheck(False)
@cython.wraparound(False)
def doubleCudaRBFFeatureGen(inputArray, outputArray, radem,
                chiArr, double betaHparam, int numFreqs, int numThreads):
    """Wraps doubleRBFFeatureGen from double_specialized_ops and uses
    it to to generate random features for an RBF kernel (this same routine
    can also be used for Matern, ARD and MiniARD). This wrapper performs all
    of the bounds checks, type checks etc and should not be bypassed.

    Args:
        inputArray (cp.ndarray): The array on which the SORF transform will be performed.
            Transform is in place so nothing is returned. Shape is (N x C).
            C must be a power of 2.
        outputArray (cp.ndarray): The output array in which the generated features will
            be stored. Must be of shape (N, numRffs) where numRffs is 2x numFreqs.
        radem (cp.ndarray): A stack of diagonal matrices of type int8_t
            of shape (3 x D x C).
        chiArr (cp.ndarray): A matrix of shape (numFreqs).
        betaHparam (double): The amplitude hyperparameter.
        numThreads (int): Not currently used, accepted only to preserve
            shared interface with CPU functions.
    """
    cdef const char *errCode
    cdef float logdim
    cdef double rbfNormConstant

    if inputArray.shape[0] == 0:
        raise ValueError("There must be at least one datapoint.")
    if inputArray.shape[1] != radem.shape[1] or inputArray.shape[2] != radem.shape[2]:
        raise ValueError("Incorrect array dims passed to a wrapped RBF feature gen function.")
    if radem.shape[0] != 3:
        raise ValueError("radem must have length 3 for dim 0.")
    if inputArray.shape[0] != outputArray.shape[0]:
        raise ValueError("inputArray and outputArray to RBF feature gen must have same number "
                    "of datapoints.")
    if outputArray.shape[1] != 2 * chiArr.shape[0]:
        raise ValueError("chiArr input to RBF feature gen is of incorrect size.")
    if 2 * inputArray.shape[1] * inputArray.shape[2] < outputArray.shape[1]:
        raise ValueError("Sizes on input and output arrays to RBF feature gen are inappropriate.")

    if not inputArray.flags["C_CONTIGUOUS"] or not radem.flags["C_CONTIGUOUS"] or not \
            chiArr.flags["C_CONTIGUOUS"] or not outputArray.flags["C_CONTIGUOUS"]:
        raise ValueError("One or more arguments to a wrapped RBF feature gen function is not "
                "C contiguous.")
    if not inputArray.dtype == "float64" or not outputArray.dtype == "float64" or \
            not chiArr.dtype == "float64":
        raise ValueError("The input, output and chiArr arrays were expected to be of type float64.")
    if not radem.dtype == "int8":
        raise ValueError("radem must be of type int8.")
    logdim = np.log2(inputArray.shape[2])
    if np.ceil(logdim) != np.floor(logdim) or inputArray.shape[2] < 2:
        raise ValueError("dim2 of the input array to RBF feature gen functions "
                            "must be a power of 2 >= 2.")

    cdef uintptr_t addr_input = inputArray.data.ptr
    cdef double *input_ptr = <double*>addr_input
    cdef uintptr_t addr_output = outputArray.data.ptr
    cdef double *output_ptr = <double*>addr_output
    cdef uintptr_t addr_chi = chiArr.data.ptr
    cdef double *chi_ptr = <double*>addr_chi
    
    cdef uintptr_t addr_radem = radem.data.ptr
    cdef int8_t *radem_ptr = <int8_t*>addr_radem

    rbfNormConstant = betaHparam * np.sqrt(1 / <double>chiArr.shape[0])

    errCode = doubleRBFFeatureGen(input_ptr, radem_ptr,
                chi_ptr, output_ptr, rbfNormConstant,
                inputArray.shape[0], inputArray.shape[1],
                inputArray.shape[2], chiArr.shape[0]);
    if errCode.decode("UTF-8") != "no_error":
        raise Exception("Fatal error encountered in doubleCudaRBFFeatureGen.")


@cython.boundscheck(False)
@cython.wraparound(False)
def floatCudaRBFFeatureGen(inputArray, outputArray, radem,
                chiArr, double betaHparam, int numFreqs, int numThreads):
    """Wraps floatRBFFeatureGen from double_specialized_ops and uses
    it to to generate random features for an RBF kernel (this same routine
    can also be used for Matern, ARD and MiniARD). This wrapper performs all
    of the bounds checks, type checks etc and should not be bypassed.

    Args:
        inputArray (cp.ndarray): The array on which the SORF transform will be performed.
            Transform is in place so nothing is returned. Shape is (N x C).
            C must be a power of 2.
        outputArray (cp.ndarray): The output array in which the generated features will
            be stored. Must be of shape (N, numRffs) where numRffs is 2x numFreqs.
        radem (cp.ndarray): A stack of diagonal matrices of type int8_t
            of shape (3 x D x C).
        chiArr (cp.ndarray): A matrix of shape (numFreqs).
        betaHparam (double): The amplitude hyperparameter.
        numThreads (int): Not currently used, accepted only to preserve
            shared interface with CPU functions.
    """
    cdef const char *errCode
    cdef float logdim
    cdef double rbfNormConstant

    if inputArray.shape[0] == 0:
        raise ValueError("There must be at least one datapoint.")
    if inputArray.shape[1] != radem.shape[1] or inputArray.shape[2] != radem.shape[2]:
        raise ValueError("Incorrect array dims passed to a wrapped RBF feature gen function.")
    if radem.shape[0] != 3:
        raise ValueError("radem must have length 3 for dim 0.")
    if inputArray.shape[0] != outputArray.shape[0]:
        raise ValueError("inputArray and outputArray to RBF feature gen must have same number "
                    "of datapoints.")
    if outputArray.shape[1] != 2 * chiArr.shape[0]:
        raise ValueError("chiArr input to RBF feature gen is of incorrect size.")
    if 2 * inputArray.shape[1] * inputArray.shape[2] < outputArray.shape[1]:
        raise ValueError("Sizes on input and output arrays to RBF feature gen are inappropriate.")

    if not inputArray.flags["C_CONTIGUOUS"] or not radem.flags["C_CONTIGUOUS"] or not \
            chiArr.flags["C_CONTIGUOUS"] or not outputArray.flags["C_CONTIGUOUS"]:
        raise ValueError("One or more arguments to a wrapped RBF feature gen function is not "
                "C contiguous.")
    if not inputArray.dtype == "float32" or not outputArray.dtype == "float64" or \
            not chiArr.dtype == "float32":
        raise ValueError("The input, output and chiArr arrays do not have expected types.")
    if not radem.dtype == "int8":
        raise ValueError("radem must be of type int8.")
    logdim = np.log2(inputArray.shape[2])
    if np.ceil(logdim) != np.floor(logdim) or inputArray.shape[2] < 2:
        raise ValueError("dim2 of the input array to RBF feature gen functions "
                            "must be a power of 2 >= 2.")

    cdef uintptr_t addr_input = inputArray.data.ptr
    cdef float *input_ptr = <float*>addr_input
    cdef uintptr_t addr_output = outputArray.data.ptr
    cdef double *output_ptr = <double*>addr_output
    cdef uintptr_t addr_chi = chiArr.data.ptr
    cdef float *chi_ptr = <float*>addr_chi
    
    cdef uintptr_t addr_radem = radem.data.ptr
    cdef int8_t *radem_ptr = <int8_t*>addr_radem

    rbfNormConstant = betaHparam * np.sqrt(1 / <double>chiArr.shape[0])

    errCode = floatRBFFeatureGen(input_ptr, radem_ptr,
                chi_ptr, output_ptr, rbfNormConstant,
                inputArray.shape[0], inputArray.shape[1],
                inputArray.shape[2], chiArr.shape[0]);
    if errCode.decode("UTF-8") != "no_error":
        raise Exception("Fatal error encountered in doubleCudaRBFFeatureGen.")
