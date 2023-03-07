"""Wraps the C functions that generate features for RBF / Matern / ARD
kernels and calculate gradients for RBF / Matern kernels.

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


cdef extern from "rbf_ops/float_rbf_ops.h" nogil:
    const char *rbfFeatureGenFloat_(float *cArray, int8_t *radem,
                float *chiArr, double *outputArray,
                double rbfNormConstant,
                int dim0, int dim1, int dim2,
                int numFreqs, int numThreads);
    const char *rbfFloatGrad_(float *cArray, int8_t *radem,
                float *chiArr, double *outputArray,
                double *gradientArray,
                double rbfNormConstant, float sigma,
                int dim0, int dim1, int dim2,
                int numFreqs, int numThreads)

cdef extern from "rbf_ops/double_rbf_ops.h" nogil:
    const char *rbfFeatureGenDouble_(double *cArray, int8_t *radem,
                double *chiArr, double *outputArray,
                double rbfNormConstant,
                int dim0, int dim1, int dim2,
                int numFreqs, int numThreads);
    const char *rbfDoubleGrad_(double *cArray, int8_t *radem,
                double *chiArr, double *outputArray,
                double *gradientArray,
                double rbfNormConstant, double sigma,
                int dim0, int dim1, int dim2,
                int numFreqs, int numThreads)


@cython.boundscheck(False)
@cython.wraparound(False)
def doubleCpuRBFFeatureGen(np.ndarray[np.float64_t, ndim=3] inputArray,
                np.ndarray[np.float64_t, ndim=2] outputArray,
                np.ndarray[np.int8_t, ndim=3] radem,
                np.ndarray[np.float64_t, ndim=1] chiArr,
                double betaHparam, int numThreads):
    """Wraps doubleRBFFeatureGen from specialized_ops and uses
    it to to generate random features for an RBF kernel (this same routine
    can also be used for Matern, ARD and MiniARD). This wrapper performs all
    of the bounds checks, type checks etc and should not be bypassed.

    Args:
        inputArray (np.ndarray): The array on which the SORF transform will be performed.
            Transform is in place so nothing is returned. Shape is (N x C).
            C must be a power of 2.
        outputArray (np.ndarray): The output array in which the generated features will
            be stored. Must be of shape (N, numRffs) where numRffs is 2x numFreqs.
        radem (np.ndarray): A stack of diagonal matrices of type int8_t
            of shape (3 x D x C).
        chiArr (np.ndarray): An array of shape (numFreqs).
        betaHparam (double): The amplitude hyperparameter.
        numThreads (int): Number of threads to run.

    Raises:
        ValueError: A ValueError is raised if unexpected or unacceptable inputs
            are supplied.
    """
    cdef const char *errCode
    cdef float logdim
    cdef double rbfNormConstant

    if inputArray.shape[0] == 0:
        raise ValueError("There must be at least one datapoint.")
    if inputArray.shape[1] != radem.shape[1] or inputArray.shape[2] != radem.shape[2]:
        raise ValueError("Incorrect array dims passed to a wrapped RBF feature gen function.")
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
    logdim = np.log2(inputArray.shape[2])
    if np.ceil(logdim) != np.floor(logdim) or inputArray.shape[2] < 2:
        raise ValueError("dim2 of the input array to RBF feature gen functions "
                            "must be a power of 2 >= 2.")


    rbfNormConstant = betaHparam * np.sqrt(1 / <double>chiArr.shape[0])

    errCode = rbfFeatureGenDouble_(&inputArray[0,0,0], &radem[0,0,0],
                &chiArr[0], &outputArray[0,0], rbfNormConstant,
                inputArray.shape[0], inputArray.shape[1],
                inputArray.shape[2], chiArr.shape[0],
                numThreads);
    if errCode.decode("UTF-8") != "no_error":
        raise Exception("Fatal error encountered in RBF feature gen.")


@cython.boundscheck(False)
@cython.wraparound(False)
def floatCpuRBFFeatureGen(np.ndarray[np.float32_t, ndim=3] inputArray,
                np.ndarray[np.float64_t, ndim=2] outputArray,
                np.ndarray[np.int8_t, ndim=3] radem,
                np.ndarray[np.float32_t, ndim=1] chiArr,
                double betaHparam, int numThreads):
    """Wraps floatRBFFeatureGen from double_specialized_ops and uses
    it to to generate random features for an RBF kernel (this same routine
    can also be used for Matern, ARD and MiniARD). This wrapper performs all
    of the bounds checks, type checks etc and should not be bypassed.

    Args:
        inputArray (np.ndarray): The array on which the SORF transform will be performed.
            Transform is in place so nothing is returned. Shape is (N x C).
            C must be a power of 2.
        outputArray (np.ndarray): The output array in which the generated features will
            be stored. Must be of shape (N, numRffs) where numRffs is 2x numFreqs.
        radem (np.ndarray): A stack of diagonal matrices of type int8_t
            of shape (3 x D x C).
        chiArr (np.ndarray): An array of shape (numFreqs).
        betaHparam (double): The amplitude hyperparameter.
        numThreads (int): Number of threads to run.

    Raises:
        ValueError: A ValueError is raised if unexpected or unacceptable inputs
            are supplied.
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

    rbfNormConstant = betaHparam * np.sqrt(1 / <double>chiArr.shape[0])

    errCode = rbfFeatureGenFloat_(&inputArray[0,0,0], &radem[0,0,0],
                &chiArr[0], &outputArray[0,0], rbfNormConstant,
                inputArray.shape[0], inputArray.shape[1],
                inputArray.shape[2], chiArr.shape[0],
                numThreads);
    if errCode.decode("UTF-8") != "no_error":
        raise Exception("Fatal error encountered in RBF feature gen.")




@cython.boundscheck(False)
@cython.wraparound(False)
def doubleCpuRBFGrad(np.ndarray[np.float64_t, ndim=3] inputArray,
                np.ndarray[np.float64_t, ndim=2] outputArray,
                np.ndarray[np.int8_t, ndim=3] radem,
                np.ndarray[np.float64_t, ndim=1] chiArr,
                double betaHparam, double sigmaHparam, int numThreads):
    """Wraps doubleRBFFeatureGen from specialized_ops and uses
    it to to generate random features for an RBF kernel (this same routine
    can also be used for Matern, ARD and MiniARD). This wrapper performs all
    of the bounds checks, type checks etc and should not be bypassed.

    Args:
        inputArray (np.ndarray): The array on which the SORF transform will be performed.
            Transform is in place so nothing is returned. Shape is (N x C).
            C must be a power of 2.
        outputArray (np.ndarray): The output array in which the generated features will
            be stored. Must be of shape (N, numRffs) where numRffs is 2x numFreqs.
        radem (np.ndarray): A stack of diagonal matrices of type int8_t
            of shape (3 x D x C).
        chiArr (np.ndarray): An array of shape (numFreqs).
        betaHparam (double): The amplitude hyperparameter.
        sigma (double): The sigma hyperparameter.
        numThreads (int): Number of threads to run.


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
    cdef np.ndarray[np.float64_t, ndim=3] gradient = np.empty((outputArray.shape[0],
                        outputArray.shape[1], 1))

    if inputArray.shape[0] == 0:
        raise ValueError("There must be at least one datapoint.")
    if inputArray.shape[1] != radem.shape[1] or inputArray.shape[2] != radem.shape[2]:
        raise ValueError("Incorrect array dims passed to a wrapped RBF feature gen function.")
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
    logdim = np.log2(inputArray.shape[2])
    if np.ceil(logdim) != np.floor(logdim) or inputArray.shape[2] < 2:
        raise ValueError("dim2 of the input array to RBF feature gen functions "
                            "must be a power of 2 >= 2.")


    rbfNormConstant = betaHparam * np.sqrt(1 / <double>chiArr.shape[0])

    errCode = rbfDoubleGrad_(&inputArray[0,0,0], &radem[0,0,0],
                &chiArr[0], &outputArray[0,0], &gradient[0,0,0],
                rbfNormConstant, sigmaHparam,
                inputArray.shape[0], inputArray.shape[1],
                inputArray.shape[2], chiArr.shape[0],
                numThreads)
    if errCode.decode("UTF-8") != "no_error":
        raise Exception("Fatal error encountered in RBF feature gen.")
    return gradient


@cython.boundscheck(False)
@cython.wraparound(False)
def floatCpuRBFGrad(np.ndarray[np.float32_t, ndim=3] inputArray,
                np.ndarray[np.float64_t, ndim=2] outputArray,
                np.ndarray[np.int8_t, ndim=3] radem,
                np.ndarray[np.float32_t, ndim=1] chiArr,
                double betaHparam, float sigmaHparam, int numThreads):
    """Wraps floatRBFFeatureGen from double_specialized_ops and uses
    it to to generate random features for an RBF kernel (this same routine
    can also be used for Matern, ARD and MiniARD). This wrapper performs all
    of the bounds checks, type checks etc and should not be bypassed.

    Args:
        inputArray (np.ndarray): The array on which the SORF transform will be performed.
            Transform is in place so nothing is returned. Shape is (N x C).
            C must be a power of 2.
        outputArray (np.ndarray): The output array in which the generated features will
            be stored. Must be of shape (N, numRffs) where numRffs is 2x numFreqs.
        radem (np.ndarray): A stack of diagonal matrices of type int8_t
            of shape (3 x D x C).
        chiArr (np.ndarray): An array of shape (numFreqs).
        betaHparam (double): The amplitude hyperparameter.
        sigmaHparam (float): The sigma hyperparameter.
        numThreads (int): Number of threads to run.

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
    cdef np.ndarray[np.float64_t, ndim=3] gradient = np.empty((outputArray.shape[0],
                        outputArray.shape[1], 1))

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

    rbfNormConstant = betaHparam * np.sqrt(1 / <double>chiArr.shape[0])

    errCode = rbfFloatGrad_(&inputArray[0,0,0], &radem[0,0,0],
                &chiArr[0], &outputArray[0,0], &gradient[0,0,0],
                rbfNormConstant, sigmaHparam,
                inputArray.shape[0], inputArray.shape[1],
                inputArray.shape[2], chiArr.shape[0],
                numThreads)
    if errCode.decode("UTF-8") != "no_error":
        raise Exception("Fatal error encountered in RBF feature gen.")
    return gradient
