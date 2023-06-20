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
from libc.stdint cimport int8_t, int32_t
from libcpp cimport bool



cdef extern from "rbf_ops/rbf_ops.h" nogil:
    const char *RBFFeatureGen[T](T cArray[], int8_t *radem,
                T chiArr[], double *outputArray,
                double rbfNormConstant,
                int dim0, int dim1, int dim2,
                int numFreqs)
    const char *RBFFeatureGrad[T](T cArray[], int8_t *radem,
                T chiArr[], double *outputArray,
                double *gradientArray, double rbfNormConstant,
                T sigma, int dim0, int dim1, int dim2,
                int numFreqs);
    const char *ardCudaGrad[T](T inputX[], double *randomFeats,
                T precompWeights[], int32_t *sigmaMap,
                double *sigmaVals, double *gradient, int dim0,\
                int dim1, int numLengthscales, int numFreqs,
                double rbfNormConstant);


@cython.boundscheck(False)
@cython.wraparound(False)
def cudaRBFFeatureGen(inputArray, outputArray, radem,
                chiArr, double betaHparam, int numThreads,
                fitIntercept = False):
    """Wraps RBFFeatureGen from double_specialized_ops and uses
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
        fitIntercept (bool): Whether to fit a y-intercept (in this case,
            the first random feature generated should be set to 1).

    Raises:
        ValueError: A ValueError is raised if unexpected or invalid inputs are supplied.
    """
    cdef const char *errCode
    cdef float logdim
    cdef double rbfNormConstant
    cdef uintptr_t addr_input = inputArray.data.ptr
    cdef uintptr_t addr_output = outputArray.data.ptr
    cdef uintptr_t addr_chi = chiArr.data.ptr
    
    cdef uintptr_t addr_radem = radem.data.ptr


    if not radem.dtype == "int8":
        raise ValueError("radem must be of type int8.")
    if not inputArray.flags["C_CONTIGUOUS"] or not radem.flags["C_CONTIGUOUS"] or not \
            chiArr.flags["C_CONTIGUOUS"] or not outputArray.flags["C_CONTIGUOUS"]:
        raise ValueError("One or more arguments to a wrapped RBF feature gen function is not "
                "C contiguous.")

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

    logdim = np.log2(inputArray.shape[2])
    if np.ceil(logdim) != np.floor(logdim) or inputArray.shape[2] < 2:
        raise ValueError("dim2 of the input array to RBF feature gen functions "
                            "must be a power of 2 >= 2.")

    if fitIntercept:
        rbfNormConstant = betaHparam * np.sqrt(2.0 / (<double>chiArr.shape[0] - 0.5))
    else:
        rbfNormConstant = betaHparam * np.sqrt(2 / <double>chiArr.shape[0])

    if inputArray.dtype == "float32" and outputArray.dtype == "float64" and \
            chiArr.dtype == "float32":
        errCode = RBFFeatureGen[float](<float*>addr_input, <int8_t*>addr_radem,
                <float*>addr_chi, <double*>addr_output, rbfNormConstant,
                inputArray.shape[0], inputArray.shape[1],
                inputArray.shape[2], chiArr.shape[0]);
    elif inputArray.dtype == "float64" and outputArray.dtype == "float64" and \
            chiArr.dtype == "float64":
        errCode = RBFFeatureGen[double](<double*>addr_input, <int8_t*>addr_radem,
                <double*>addr_chi, <double*>addr_output, rbfNormConstant,
                inputArray.shape[0], inputArray.shape[1],
                inputArray.shape[2], chiArr.shape[0]);
    else:
        raise ValueError("The input and chiArr arrays are of inconsistent types.")

    if errCode.decode("UTF-8") != "no_error":
        raise Exception("Fatal error encountered in CudaRBFFeatureGen.")
    if fitIntercept:
        outputArray[:,0] = betaHparam



@cython.boundscheck(False)
@cython.wraparound(False)
def cudaRBFGrad(inputArray, outputArray, radem,
                chiArr, double betaHparam, double sigmaHparam,
                int numThreads, bool fitIntercept = False):
    """Wraps RBFFeatureGen and uses
    it to to generate random features for an RBF kernel
    together with the gradient. This wrapper performs all
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
        fitIntercept (bool): Whether to fit a y-intercept (in this case,
            the first random feature generated should be set to 1).

    Raises:
        ValueError: A ValueError is raised if unexpected or invalid inputs are supplied.

    Returns:
        gradient (cp.ndarray): An array containing the gradient values.
    """
    cdef const char *errCode
    cdef float logdim
    cdef double rbfNormConstant
    cdef uintptr_t addr_input = inputArray.data.ptr
    cdef uintptr_t addr_output = outputArray.data.ptr
    cdef uintptr_t addr_chi = chiArr.data.ptr
    
    cdef uintptr_t addr_radem = radem.data.ptr

    gradient = cp.zeros((outputArray.shape[0], outputArray.shape[1], 1),
            dtype=cp.float64)
    cdef uintptr_t addr_gradient = gradient.data.ptr

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
    if not radem.dtype == "int8":
        raise ValueError("radem must be of type int8.")
    logdim = np.log2(inputArray.shape[2])
    if np.ceil(logdim) != np.floor(logdim) or inputArray.shape[2] < 2:
        raise ValueError("dim2 of the input array to RBF feature gen functions "
                            "must be a power of 2 >= 2.")


    if fitIntercept:
        rbfNormConstant = betaHparam * np.sqrt(2.0 / (<double>chiArr.shape[0] - 0.5))
    else:
        rbfNormConstant = betaHparam * np.sqrt(2 / <double>chiArr.shape[0])

    if inputArray.dtype == "float32" and outputArray.dtype == "float64" and \
            chiArr.dtype == "float32":
        errCode = RBFFeatureGrad[float](<float*>addr_input, <int8_t*>addr_radem,
                <float*>addr_chi, <double*>addr_output, <double*>addr_gradient,
                rbfNormConstant, sigmaHparam,
                inputArray.shape[0], inputArray.shape[1],
                inputArray.shape[2], chiArr.shape[0]);
    elif inputArray.dtype == "float64" and outputArray.dtype == "float64" and \
            chiArr.dtype == "float64":
        errCode = RBFFeatureGrad[double](<double*>addr_input, <int8_t*>addr_radem,
                <double*>addr_chi, <double*>addr_output, <double*>addr_gradient,
                rbfNormConstant, sigmaHparam,
                inputArray.shape[0], inputArray.shape[1],
                inputArray.shape[2], chiArr.shape[0]);
    else:    
        raise ValueError("The input and chiArr arrays are of inconsistent types.")
    if errCode.decode("UTF-8") != "no_error":
        raise Exception("Fatal error encountered in doubleCudaRBFFeatureGen.")
    if fitIntercept:
        outputArray[:,0] = betaHparam
        gradient[:,0] = 0
    return gradient





@cython.boundscheck(False)
@cython.wraparound(False)
def cudaMiniARDGrad(inputX, outputArray, precompWeights,
                sigmaMap, sigmaVals, double betaHparam, int numThreads,
                bool fitIntercept = False):
    """Performs gradient calculations for the MiniARD kernel, using
    pregenerated features and precomputed weights.

    Args:
        inputX (np.ndarray): The original input data.
        outputArray (np.ndarray): The random features generated using the FHT-
            based procedure.
        precompWeights (np.ndarray): The FHT-rf gen procedure applied to an
            identity matrix.
        sigmaMap (np.ndarray): An array mapping which lengthscales correspond
            to which positions in the input.
        sigmaVals (cp.ndarray): The lengthscale values, in an array of the same
            dimensionality as the input.
        betaHparam (double): The amplitude hyperparameter.
        numThreads (int): Number of threads to run.
        fitIntercept (bool): Whether to fit a y-intercept (in this case,
            the first random feature generated should be set to 1).

    Raises:
        ValueError: A ValueError is raised if unexpected or unacceptable inputs
            are supplied.

    Returns:
        gradient (np.ndarray): An array of shape (N x 2 * numFreqs x 1) containing
            the gradient w/r/t sigma.
    """
    cdef const char *errCode
    cdef float logdim
    cdef double rbfNormConstant
    cdef int numLengthscales = sigmaMap.max() + 1
    gradient = cp.zeros((outputArray.shape[0],
                        outputArray.shape[1], numLengthscales))

    if len(inputX.shape) != 2 or len(outputArray.shape) != 2 or \
            len(precompWeights.shape) != 2 or len(sigmaMap.shape) != 1:
        raise ValueError("The input arrays to a wrapped RBF feature gen function have incorrect "
                "shapes.")

    if inputX.shape[0] == 0 or inputX.shape[1] == 0:
        raise ValueError("There must be at least one datapoint.")
    if inputX.shape[0] != outputArray.shape[0] or precompWeights.shape[1] != inputX.shape[1]:
        raise ValueError("Incorrect array dims passed to a wrapped RBF "
                    "feature gen function.")
    if outputArray.shape[1] != 2 * precompWeights.shape[0] or sigmaMap.shape[0] != \
            precompWeights.shape[1] or sigmaVals.shape[0] != sigmaMap.shape[0]:
        raise ValueError("Incorrect array dims passed to a wrapped RBF "
                    "feature gen function.")

    if not inputX.flags["C_CONTIGUOUS"] or not outputArray.flags["C_CONTIGUOUS"] or not \
            precompWeights.flags["C_CONTIGUOUS"] or not sigmaMap.flags["C_CONTIGUOUS"] or \
            not sigmaVals.flags["C_CONTIGUOUS"]:
        raise ValueError("One or more arguments to a wrapped RBF feature gen function is not "
                "C contiguous.")


    cdef uintptr_t addr_input = inputX.data.ptr
    cdef uintptr_t addr_random_feats = outputArray.data.ptr

    cdef uintptr_t addr_sigma_map = sigmaMap.data.ptr
    cdef int32_t *sigmaMap_ptr = <int32_t*>addr_sigma_map
    cdef uintptr_t addr_sigma_vals = sigmaVals.data.ptr

    cdef uintptr_t addr_grad = gradient.data.ptr
    cdef uintptr_t addr_precomp_weights = precompWeights.data.ptr

    if fitIntercept:
        rbfNormConstant = betaHparam * np.sqrt(2.0 / (<double>precompWeights.shape[0] - 0.5))
    else:
        rbfNormConstant = betaHparam * np.sqrt(2 / <double>precompWeights.shape[0])

    if inputX.dtype == "float32" and precompWeights.dtype == "float32" and \
            outputArray.dtype == "float64" and sigmaMap.dtype == "int32" and \
            sigmaVals.dtype == "float64":
        errCode = ardCudaGrad[float](<float*>addr_input, <double*>addr_random_feats,
                <float*>addr_precomp_weights, <int32_t*>addr_sigma_map,
                <double*>addr_sigma_vals,
                <double*>addr_grad, inputX.shape[0], inputX.shape[1],
                gradient.shape[2], precompWeights.shape[0],
                rbfNormConstant)

    elif inputX.dtype == "float64" and precompWeights.dtype == "float64" and \
            outputArray.dtype == "float64" and sigmaMap.dtype == "int32" and \
            sigmaVals.dtype == "float64":
        errCode = ardCudaGrad[double](<double*>addr_input, <double*>addr_random_feats,
                <double*>addr_precomp_weights, <int32_t*>addr_sigma_map,
                <double*>addr_sigma_vals,
                <double*>addr_grad, inputX.shape[0], inputX.shape[1],
                gradient.shape[2], precompWeights.shape[0],
                rbfNormConstant)
    else:
        raise ValueError("The input arrays to a wrapped RBF feature gen function have incorrect "
                "types.")

    if errCode.decode("UTF-8") != "no_error":
        raise Exception("Fatal error encountered in RBF feature gen.")
    if fitIntercept:
        outputArray[:,0] = betaHparam
        gradient[:,0,:] = 0
    return gradient
