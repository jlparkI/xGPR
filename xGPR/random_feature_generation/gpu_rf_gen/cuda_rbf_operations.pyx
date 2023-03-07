"""Handles RBF / Matern / ARD feature generation and gradients.

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



cdef extern from "rbf_ops/double_rbf_ops.h" nogil:
    const char *doubleRBFFeatureGen(double *cArray, int8_t *radem,
                double *chiArr, double *outputArray,
                double rbfNormConstant,
                int dim0, int dim1, int dim2,
                int numFreqs)
    const char *doubleRBFFeatureGrad(double *cArray, int8_t *radem,
                double *chiArr, double *outputArray,
                double *gradientArray, double rbfNormConstant,
                double sigma, int dim0, int dim1, int dim2,
                int numFreqs);

cdef extern from "rbf_ops/float_rbf_ops.h" nogil:
    const char *floatRBFFeatureGen(float *cArray, int8_t *radem,
                float *chiArr, double *outputArray,
                double rbfNormConstant,
                int dim0, int dim1, int dim2,
                int numFreqs)
    const char *floatRBFFeatureGrad(float *cArray, int8_t *radem,
                float *chiArr, double *outputArray,
                double *gradientArray, double rbfNormConstant,
                float sigma, int dim0, int dim1, int dim2,
                int numFreqs);


@cython.boundscheck(False)
@cython.wraparound(False)
def doubleCudaRBFFeatureGen(inputArray, outputArray, radem,
                chiArr, double betaHparam, int numThreads):
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

    Raises:
        ValueError: A ValueError is raised if unexpected or invalid inputs are supplied.
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
        raise Exception("Fatal error encountered in CudaRBFFeatureGen.")


@cython.boundscheck(False)
@cython.wraparound(False)
def floatCudaRBFFeatureGen(inputArray, outputArray, radem,
                chiArr, double betaHparam, int numThreads):
    """Wraps floatRBFFeatureGen from double_specialized_ops and uses
    it to to generate random features for an RBF kernel (this same routine
    can also be used for Matern, ARD and MiniARD). This wrapper performs all
    of the bounds checks, type checks etc and should not be bypassed.

    Args:
        inputArray (cp.ndarray): The array on which the SORF transform will be performed.
            Shape is (N x C). C must be a power of 2.
        outputArray (cp.ndarray): The output array in which the generated features will
            be stored. Must be of shape (N, numRffs) where numRffs is 2x numFreqs.
        radem (cp.ndarray): A stack of diagonal matrices of type int8_t
            of shape (3 x D x C).
        chiArr (cp.ndarray): A matrix of shape (numFreqs).
        betaHparam (double): The amplitude hyperparameter.
        numThreads (int): Not currently used, accepted only to preserve
            shared interface with CPU functions.

    Raises:
        ValueError: A ValueError is raised if unexpected or invalid inputs are supplied.
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
        raise Exception("Fatal error encountered in CudaRBFFeatureGen.")






@cython.boundscheck(False)
@cython.wraparound(False)
def doubleCudaRBFGrad(inputArray, outputArray, radem,
                chiArr, double betaHparam, double sigmaHparam,
                int numThreads):
    """Wraps doubleRBFFeatureGen from double_specialized_ops and uses
    it to to generate random features for an RBF kernel (this same routine
    can also be used for Matern, ARD and MiniARD). This wrapper performs all
    of the bounds checks, type checks etc and should not be bypassed.

    Args:
        inputArray (cp.ndarray): The array on which the SORF transform will be performed.
            Shape is (N x C). C must be a power of 2.
        outputArray (cp.ndarray): The output array in which the generated features will
            be stored. Must be of shape (N, numRffs) where numRffs is 2x numFreqs.
        radem (cp.ndarray): A stack of diagonal matrices of type int8_t
            of shape (3 x D x C).
        chiArr (cp.ndarray): A matrix of shape (numFreqs).
        betaHparam (double): The amplitude hyperparameter.
        sigmaHparam (double): The sigma hyperparameter.
        numThreads (int): Not currently used, accepted only to preserve
            shared interface with CPU functions.

    Raises:
        ValueError: A ValueError is raised if unexpected or invalid inputs are supplied.

    Returns:
        gradient (cp.ndarray): An array containing the gradient values.
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

    gradient = cp.zeros((outputArray.shape[0], outputArray.shape[1], 1),
            dtype=cp.float64)
    cdef uintptr_t addr_gradientArray = gradient.data.ptr
    cdef double *gradient_ptr = <double*>addr_gradientArray

    rbfNormConstant = betaHparam * np.sqrt(1 / <double>chiArr.shape[0])

    errCode = doubleRBFFeatureGrad(input_ptr, radem_ptr,
                chi_ptr, output_ptr, gradient_ptr,
                rbfNormConstant, sigmaHparam,
                inputArray.shape[0], inputArray.shape[1],
                inputArray.shape[2], chiArr.shape[0]);
    if errCode.decode("UTF-8") != "no_error":
        raise Exception("Fatal error encountered in doubleCudaRBFFeatureGen.")
    return gradient

@cython.boundscheck(False)
@cython.wraparound(False)
def floatCudaRBFGrad(inputArray, outputArray, radem,
                chiArr, double betaHparam, float sigmaHparam,
                int numThreads):
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
        sigmaHparam (float): The sigma hyperparameter.
        numThreads (int): Not currently used, accepted only to preserve
            shared interface with CPU functions.

    Raises:
        ValueError: A ValueError is raised if unexpected or invalid inputs are supplied.

    Returns:
        gradient (cp.ndarray): An array containing the gradient values.
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

    gradient = cp.zeros((outputArray.shape[0], outputArray.shape[1], 1),
            dtype=cp.float64)
    cdef uintptr_t addr_gradientArray = gradient.data.ptr
    cdef double *gradient_ptr = <double*>addr_gradientArray

    rbfNormConstant = betaHparam * np.sqrt(1 / <double>chiArr.shape[0])

    errCode = floatRBFFeatureGrad(input_ptr, radem_ptr,
                chi_ptr, output_ptr, gradient_ptr,
                rbfNormConstant, sigmaHparam,
                inputArray.shape[0], inputArray.shape[1],
                inputArray.shape[2], chiArr.shape[0]);
    if errCode.decode("UTF-8") != "no_error":
        raise Exception("Fatal error encountered in doubleCudaRBFFeatureGen.")
    return gradient
