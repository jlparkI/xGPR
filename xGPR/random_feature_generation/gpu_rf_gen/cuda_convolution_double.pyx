"""Handles hadamard transform-based convolution operations for sequences and graphs
if the input array is an array of doubles.

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
from libc.stdint cimport int8_t, int32_t


cdef extern from "convolution_ops/convolution.h" nogil:
    const char *doubleConv1dPrep(int8_t *radem,
                double *reshapedx, int reshapeddim0, 
                int reshapeddim1, int reshapeddim2,
                int startposition, int numfreqs)

cdef extern from "convolution_ops/rbf_convolution.h" nogil:
    const char *doubleConvRBFFeatureGen(int8_t *radem, double *reshapedX,
                double *featureArray, double *chiArr, double *outputArray,
                int reshapedDim0, int reshapedDim1, int reshapedDim2,
                int numFreqs, int rademShape2, double scalingTerm);
    const char *doubleConvRBFFeatureGrad(int8_t *radem, double *reshapedX,
                double *featureArray, double *chiArr, double *outputArray,
                double *gradientArray, double sigma,
                int reshapedDim0, int reshapedDim1, int reshapedDim2,
                int numFreqs, int rademShape2, double scalingTerm);


cdef extern from "convolution_ops/ard_convolution.h" nogil:
    const char *ardConvCudaDoubleGrad(double *inputX, double *randomFeats,
                double *precompWeights, int32_t *sigmaMap,
                double *sigmaVals, double *gradient, int dim0,
                int dim1, int dim2, int numLengthscales,
                int numFreqs, double rbfNormConstant);



@cython.boundscheck(False)
@cython.wraparound(False)
def doubleGpuConv1dMaxpool(reshapedX, radem, outputArray, chiArr,
                int numThreads, bint subtractMean = False):
    """Uses wrapped C extensions to perform random feature generation
    with ReLU activation and maxpooling.

    Args:
        reshapedX (cp.ndarray): An array of type float64 from which
            the features will be generated. Is not modified. Must
            be of shape (N x D x C) where C is a power of 2. Should
            have been reshaped to be appropriate for convolution.
        radem (cp.ndarray): A stack of diagonal matrices of type int8_t
            of shape (3 x 1 x m * C), where R is the number of random
            features requested and m is ceil(R / C).
        outputArray (cp.ndarray): An N x R array in which the output features
            will be stored.
        chiArr (cp.ndarray): A stack of diagonal matrices stored as an
            array of shape (R) drawn from a chi distribution.
        num_threads (int): This argument is so that this function has
            the same interface as the CPU SORF Transform. It is not
            needed for the GPU transform and is ignored.
        subtractMean (bool): If True, subtract the mean of each row from
            that row.
    """
    cdef const char *errCode
    cdef int i, startPosition, cutoff
    cdef int num_repeats = (radem.shape[2] + reshapedX.shape[2] - 1) // reshapedX.shape[2]
    reshapedXCopy = reshapedX.copy()
    
    if reshapedX.shape[0] == 0:
        raise ValueError("There must be at least one datapoint.")
    if reshapedX.shape[0] != outputArray.shape[0]:
        raise ValueError("The number of input and output datapoints do not "
                "agree.")
    if not len(radem.shape) == 3:
        raise ValueError("radem must be a 3d array.")
    if not len(chiArr.shape) == 1 or not len(outputArray.shape) == 2:
        raise ValueError("chiArr must be a 1d array; outputArray must be 2d.")
    if not len(reshapedX.shape) == 3:
        raise ValueError("X must be a 3d array.")
    if not radem.dtype == "int8":
        raise ValueError("radem must be int8.")
    if not outputArray.dtype == "float64" or not reshapedX.dtype == "float64":
        raise ValueError("reshapedX, outputArray must be float64.")
    if not chiArr.dtype == "float64":
        raise ValueError("chiArr must be float64.")


    if radem.shape[0] != 3 or radem.shape[1] != 1:
        raise ValueError("radem must have length 3 for dim 0 and length 1 for dim1.")
    if outputArray.shape[1] != radem.shape[2] or chiArr.shape[0] != radem.shape[2]:
        raise ValueError("Array shapes not correct.")
        
    logdim = np.log2(reshapedX.shape[2])
    if np.ceil(logdim) != np.floor(logdim) or reshapedX.shape[2] < 2:
        raise ValueError("Array shapes not correct.")
    if not radem.shape[2] % reshapedX.shape[2] == 0:
        raise ValueError("Array shapes not correct.")

    #Make sure that all inputs are C-contiguous.
    if not outputArray.flags["C_CONTIGUOUS"] or not reshapedX.flags["C_CONTIGUOUS"] \
            or not radem.flags["C_CONTIGUOUS"] or not chiArr.flags["C_CONTIGUOUS"]:
        raise ValueError("One or more arguments is not C contiguous.")


    cdef uintptr_t addr_reshapedCopy = reshapedXCopy.data.ptr
    cdef double *reshapedXCopyPtr = <double*>addr_reshapedCopy
    
    cdef uintptr_t addr_radem = radem.data.ptr
    cdef int8_t *radem_ptr = <int8_t*>addr_radem
    
    startPosition, cutoff = 0, reshapedX.shape[2]
    
    for i in range(num_repeats):
        reshapedXCopy[:] = reshapedX
        errCode = doubleConv1dPrep(radem_ptr,
                    reshapedXCopyPtr, reshapedX.shape[0], reshapedX.shape[1], 
                    reshapedX.shape[2], i * reshapedX.shape[2],
                    radem.shape[2])
        if errCode.decode("UTF-8") != "no_error":
            raise Exception("Fatal error encountered in doubleGpuConv1dTransform_.")
        
        reshapedXCopy *= chiArr[None,None,(i * reshapedX.shape[2]):((i+1) * reshapedX.shape[2])]
        outputArray[:,startPosition:cutoff] = reshapedXCopy.max(axis=1)
        if subtractMean:
            outputArray[:,startPosition:cutoff] -= reshapedXCopy.mean(axis=1)

        cutoff += reshapedX.shape[2]
        startPosition += reshapedX.shape[2]



@cython.boundscheck(False)
@cython.wraparound(False)
def doubleGpuConv1dFGen(reshapedX, radem, outputArray, chiArr,
                int numThreads, double beta_):
    """Uses wrapped C functions to generate random features for FHTConv1d, GraphConv1d,
    and related kernels. This function cannot be used to calculate the gradient
    so is only used for forward pass only (during fitting, inference, non-gradient-based
    optimization). It does not multiply by the lengthscales, so caller should do this.
    (This enables this function to also be used for GraphARD kernels if desired.)

    Args:
        reshapedX (cp.ndarray): Raw data reshaped so that a convolution can be
            performed on it using orthogonal random features with the SORF
            operation. This array is not modified in place -- rather the features
            that are generated are stored in outputArray. Shape is (N x D x C) for 
            N datapoints. C must be a power of 2.
        radem (cp.ndarray): A stack of diagonal matrices with elements drawn from the
            Rademacher distribution. Shape must be (3 x D x C).
        outputArray (cp.ndarray): A numpy array in which the generated features will be
            stored. Is modified in-place.
        chiArr (cp.ndarray): A stack of diagonal matrices drawn from a chi distribution.
        num_threads (int): Number of threads to use for FHT.
        beta (double): The amplitude.

    Raises:
        ValueError: A ValueError is raised if unexpected or invalid inputs are supplied.
    """
    cdef const char *errCode
    cdef double scalingTerm

    if len(chiArr.shape) != 1 or len(radem.shape) != 3 or len(reshapedX.shape) != 3:
        raise ValueError("chiArr should be a 1d array. radem and reshapedX should be 3d arrays.")
    if len(outputArray.shape) != 2:
        raise ValueError("outputArray should be a 2d array.")

    if reshapedX.shape[0] == 0 or reshapedX.shape[1] == 0:
        raise ValueError("There must be at least one datapoint.")
    if reshapedX.shape[0] != outputArray.shape[0]:
        raise ValueError("The number of datapoints in the outputs and the inputs do "
                "not agree.")
    if radem.shape[0] != 3 or radem.shape[1] != 1:
        raise ValueError("radem must have length 3 for dim 0 and length 1 for dim1.")

    if outputArray.shape[1] % 2 != 0 or outputArray.shape[1] < 2:
        raise ValueError("Shape of output array is not appropriate.")
    
    if 2 * chiArr.shape[0] != outputArray.shape[1] or chiArr.shape[0] > radem.shape[2]:
        raise ValueError("Shape of output array and / or chiArr is inappropriate.")

    logdim = np.log2(reshapedX.shape[2])
    if np.ceil(logdim) != np.floor(logdim) or reshapedX.shape[2] < 2:
        raise ValueError("dim2 of the reshapedX array must be a power of 2 >= 2.")
    if not radem.shape[2] % reshapedX.shape[2] == 0:
        raise ValueError("The number of sampled frequencies should be an integer multiple of "
                "reshapedX.shape[2].")
    
    if not radem.dtype == "int8":
        raise ValueError("radem must be int8.")
    if not outputArray.dtype == "float64" or not reshapedX.dtype == "float64":
        raise ValueError("reshapedX, outputArray must be float64.")
    if not chiArr.dtype == "float64":
        raise ValueError("chiArr must be float64.")

    if not outputArray.flags["C_CONTIGUOUS"] or not reshapedX.flags["C_CONTIGUOUS"] or not radem.flags["C_CONTIGUOUS"] \
        or not chiArr.flags["C_CONTIGUOUS"]:
        raise ValueError("One or more arguments is not C contiguous.")

    featureArray = cp.zeros((reshapedX.shape[0], reshapedX.shape[1], reshapedX.shape[2]),
                            dtype = cp.float64)

    cdef uintptr_t addr_reshapedX = reshapedX.data.ptr
    cdef double *reshapedXPtr = <double*>addr_reshapedX
    cdef uintptr_t addr_featureArray = featureArray.data.ptr
    cdef double *featureArrayPtr = <double*>addr_featureArray
 
    cdef uintptr_t addr_radem = radem.data.ptr
    cdef int8_t *radem_ptr = <int8_t*>addr_radem
    cdef uintptr_t addr_chi = chiArr.data.ptr
    cdef double *chiArrPtr = <double*>addr_chi

    cdef uintptr_t addr_outputArray = outputArray.data.ptr
    cdef double *outputArrayPtr = <double*>addr_outputArray


    scalingTerm = np.sqrt(1 / <double>chiArr.shape[0])
    scalingTerm *= beta_

    errCode = doubleConvRBFFeatureGen(radem_ptr, reshapedXPtr,
                    featureArrayPtr, chiArrPtr, outputArrayPtr,
                    reshapedX.shape[0], reshapedX.shape[1],
                    reshapedX.shape[2], chiArr.shape[0],
                    radem.shape[2], scalingTerm)

    if errCode.decode("UTF-8") != "no_error":
        raise Exception("Fatal error encountered while performing FHT RF generation.")




@cython.boundscheck(False)
@cython.wraparound(False)
def doubleGpuConvGrad(reshapedX, radem, outputArray, chiArr,
                int numThreads, double sigma, double beta_):
    """Performs feature generation for RBF-based convolution kernels while also performing
    gradient calculations.

    Args:
        reshapedX (cp.ndarray): Raw data reshaped so that a convolution can be
            performed on it using orthogonal random features with the SORF
            operation. This array is not modified in place -- rather the features
            that are generated are stored in outputArray. Shape is (N x D x C) for 
            N datapoints. C must be a power of 2.
        radem (cp.ndarray): A stack of diagonal matrices with elements drawn from the
            Rademacher distribution. Shape must be (3 x D x C).
        outputArray (cp.ndarray): A numpy array in which the generated features will be
            stored. Is modified in-place.
        chiArr (cp.ndarray): A stack of diagonal matrices drawn from a chi distribution.
        num_threads (int): Number of threads to use for FHT.
        sigma (double): The lengthscale.
        beta (double): The amplitude.

    Raises:
        ValueError: A ValueError is raised if unexpected or invalid inputs are supplied.

    Returns:
        gradient (cp.ndarray); An array of shape output.shape[0] x output.shape[1] x 1.
    """
    cdef const char *errCode
    cdef double scalingTerm

    if len(chiArr.shape) != 1 or len(radem.shape) != 3 or len(reshapedX.shape) != 3:
        raise ValueError("chiArr should be a 1d array. radem and reshapedX should be 3d arrays.")
    if len(outputArray.shape) != 2:
        raise ValueError("outputArray should be a 2d array.")

    if reshapedX.shape[0] == 0 or reshapedX.shape[1] == 0:
        raise ValueError("There must be at least one datapoint.")
    if reshapedX.shape[0] != outputArray.shape[0]:
        raise ValueError("The number of datapoints in the outputs and the inputs do "
                "not agree.")
    if radem.shape[0] != 3 or radem.shape[1] != 1:
        raise ValueError("radem must have length 3 for dim 0 and length 1 for dim1.")

    if outputArray.shape[1] % 2 != 0 or outputArray.shape[1] < 2:
        raise ValueError("Shape of output array is not appropriate.")
    
    if 2 * chiArr.shape[0] != outputArray.shape[1] or chiArr.shape[0] > radem.shape[2]:
        raise ValueError("Shape of output array and / or chiArr is inappropriate.")
    
    logdim = np.log2(reshapedX.shape[2])
    if np.ceil(logdim) != np.floor(logdim) or reshapedX.shape[2] < 2:
        raise ValueError("dim2 of the reshapedX array must be a power of 2 >= 2.")
    if not radem.shape[2] % reshapedX.shape[2] == 0:
        raise ValueError("The number of sampled frequencies should be an integer multiple of "
                "reshapedX.shape[2].")
    
    if not radem.dtype == "int8":
        raise ValueError("radem must be int8.")
    if not outputArray.dtype == "float64" or not reshapedX.dtype == "float64":
        raise ValueError("reshapedX, outputArray must be float64.")
    if not chiArr.dtype == "float64":
        raise ValueError("chiArr must be float64.")

    if not outputArray.flags["C_CONTIGUOUS"] or not reshapedX.flags["C_CONTIGUOUS"] or not radem.flags["C_CONTIGUOUS"] \
        or not chiArr.flags["C_CONTIGUOUS"]:
        raise ValueError("One or more arguments is not C contiguous.")


    gradient = cp.zeros((outputArray.shape[0], outputArray.shape[1], 1), dtype=cp.float64)

    featureArray = cp.zeros((reshapedX.shape[0], reshapedX.shape[1], reshapedX.shape[2]),
                            dtype = cp.float64)

    cdef uintptr_t addr_reshapedX = reshapedX.data.ptr
    cdef double *reshapedXPtr = <double*>addr_reshapedX
    cdef uintptr_t addr_featureArray = featureArray.data.ptr
    cdef double *featureArrayPtr = <double*>addr_featureArray
 
    cdef uintptr_t addr_radem = radem.data.ptr
    cdef int8_t *radem_ptr = <int8_t*>addr_radem
    cdef uintptr_t addr_chi = chiArr.data.ptr
    cdef double *chiArrPtr = <double*>addr_chi

    cdef uintptr_t addr_outputArray = outputArray.data.ptr
    cdef double *outputArrayPtr = <double*>addr_outputArray
    cdef uintptr_t addr_gradientArray = gradient.data.ptr
    cdef double *gradientArrayPtr = <double*>addr_gradientArray


    scalingTerm = np.sqrt(1 / <double>chiArr.shape[0])
    scalingTerm *= beta_
    
    errCode = doubleConvRBFFeatureGrad(radem_ptr, reshapedXPtr,
                featureArrayPtr, chiArrPtr, outputArrayPtr,
                gradientArrayPtr, sigma,
                reshapedX.shape[0], reshapedX.shape[1], reshapedX.shape[2],
                chiArr.shape[0], radem.shape[2], scalingTerm);
    if errCode.decode("UTF-8") != "no_error":
        raise Exception("Fatal error encountered while performing FHT RF generation.")
    return gradient



@cython.boundscheck(False)
@cython.wraparound(False)
def doubleGpuGraphMiniARDGrad(inputX, randomFeats,
                precompWeights, sigmaMap, sigmaVals,
                double betaHparam, int numThreads):
    """Performs gradient calculations for the GraphMiniARD kernel, using
    pregenerated features and precomputed weights.

    Args:
        inputX (array): The original input data.
        randomFeats (array): The random features generated using the FHT-
            based procedure; will be modified in place.
        precompWeights (array): The FHT-rf gen procedure applied to an
            identity matrix.
        sigmaMap (array): An array mapping which lengthscales correspond
            to which positions in the input.
        sigmaVals (array): The lengthscale values, in an array of the same
            dimensionality as the input.
        betaHparam (double): The amplitude hyperparameter.
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
    cdef int numLengthscales = sigmaMap.max() + 1
    gradient = cp.zeros((randomFeats.shape[0],
                        randomFeats.shape[1], numLengthscales))

    if len(inputX.shape) != 3 or len(randomFeats.shape) != 2 or \
            len(precompWeights.shape) != 2 or len(sigmaMap.shape) != 1:
        raise ValueError("The input arrays to a wrapped RBF feature gen function have incorrect "
                "shapes.")

    if inputX.shape[0] == 0 or inputX.shape[1] == 0:
        raise ValueError("There must be at least one datapoint.")
    if inputX.shape[0] != randomFeats.shape[0] or precompWeights.shape[1] != inputX.shape[2]:
        raise ValueError("Incorrect array dims passed to a wrapped RBF "
                    "feature gen function.")
    if randomFeats.shape[1] != 2 * precompWeights.shape[0] or sigmaMap.shape[0] != \
            precompWeights.shape[1] or sigmaVals.shape[0] != sigmaMap.shape[0]:
        raise ValueError("Incorrect array dims passed to a wrapped RBF "
                    "feature gen function.")

    if not inputX.flags["C_CONTIGUOUS"] or not randomFeats.flags["C_CONTIGUOUS"] or not \
            precompWeights.flags["C_CONTIGUOUS"] or not sigmaMap.flags["C_CONTIGUOUS"] or \
            not sigmaVals.flags["C_CONTIGUOUS"]:
        raise ValueError("One or more arguments to a wrapped RBF feature gen function is not "
                "C contiguous.")

    if not inputX.dtype == "float64" or not precompWeights.dtype == "float64" or \
            not randomFeats.dtype == "float64" or not sigmaMap.dtype == "int32" or \
            not sigmaVals.dtype == "float64":
        raise ValueError("The input arrays to a wrapped RBF feature gen function have incorrect "
                "types.")

    cdef uintptr_t addr_input = inputX.data.ptr
    cdef double *inputX_ptr = <double*>addr_input
    cdef uintptr_t addr_random_feats = randomFeats.data.ptr
    cdef double *randomFeats_ptr = <double*>addr_random_feats

    cdef uintptr_t addr_sigma_map = sigmaMap.data.ptr
    cdef int32_t *sigmaMap_ptr = <int32_t*>addr_sigma_map
    cdef uintptr_t addr_sigma_vals = sigmaVals.data.ptr
    cdef double *sigmaVals_ptr = <double*>addr_sigma_vals

    cdef uintptr_t addr_grad = gradient.data.ptr
    cdef double *gradient_ptr = <double*>addr_grad
    cdef uintptr_t addr_precomp_weights = precompWeights.data.ptr
    cdef double *precompWeights_ptr = <double*>addr_precomp_weights

    rbfNormConstant = betaHparam * np.sqrt(1 / <double>precompWeights.shape[0])

    errCode = ardConvCudaDoubleGrad(inputX_ptr, randomFeats_ptr,
                precompWeights_ptr, sigmaMap_ptr, sigmaVals_ptr,
                gradient_ptr, inputX.shape[0], inputX.shape[1],
                inputX.shape[2], gradient.shape[2],
                precompWeights.shape[0], rbfNormConstant)
    if errCode.decode("UTF-8") != "no_error":
        raise Exception("Fatal error encountered in RBF feature gen.")
    return gradient
