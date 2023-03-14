"""Wraps the C functions that generate random features for convolution
kernels & related kernels, for single-precision arithmetic.
Also performs all of the bounds and safety checks needed to use these
functions (the C functions do not do their own bounds checking). It
is EXTREMELY important that this wrapper not be bypassed for this
reason -- it double checks all of the array dimensions, types,
is data contiguous etc. before calling the wrapped C functions."""
import numpy as np
cimport numpy as np
cimport cython
from libc cimport stdint
from libc.stdint cimport int8_t, int32_t
import math



cdef extern from "convolution_ops/conv1d_operations.h" nogil:
    const char *floatConv1dPrep_(int8_t *radem, float *reshapedX,
            int numThreads, int reshapedDim0,
            int reshapedDim1, int reshapedDim2,
            int startPosition, int numFreqs)

cdef extern from "convolution_ops/rbf_convolution.h" nogil:
    const char *floatConvRBFFeatureGen_(int8_t *radem, float *reshapedX,
            float *copyBuffer, float *chiArr, double *outputArray,
            int numThreads, int reshapedDim0,
            int reshapedDim1, int reshapedDim2,
            int numFreqs, int rademShape2)
    const char *floatConvRBFGrad_(int8_t *radem, float *reshapedX,
            float *copyBuffer, float *chiArr, double *outputArray,
            double *gradientArray, float sigma,
            int numThreads, int reshapedDim0,
            int reshapedDim1, int reshapedDim2,
            int numFreqs, int rademShape2)

    
cdef extern from "convolution_ops/ard_convolution.h" nogil:
    const char *graphARDFloatGrad_(float *inputX, double *randomFeatures,
            float *precompWeights, int32_t *sigmaMap, double *sigmaVals,
            double *gradient, int dim0, int dim1, int dim2,
            int numLengthscales, int numFreqs, double rbfNormConstant,
            int numThreads);


@cython.boundscheck(False)
@cython.wraparound(False)
def floatCpuConv1dMaxpool(np.ndarray[np.float32_t, ndim=3] reshapedX,
                np.ndarray[np.int8_t, ndim=3] radem,
                np.ndarray[np.float64_t, ndim=2] outputArray,
                np.ndarray[np.float32_t, ndim=1] chiArr,
                int numThreads, bint subtractMean = False):
    """Uses wrapped C extensions to perform random feature generation
    with ReLU activation and maxpooling.

    Args:
        reshapedX (np.ndarray): An array of type float32 from which
            the features will be generated. Is not modified. Must
            be of shape (N x D x C) where C is a power of 2. Should
            have been reshaped to be appropriate for convolution.
        radem (np.ndarray): A stack of diagonal matrices of type int8_t
            of shape (3 x 1 x m * C), where R is the number of random
            features requested and m is ceil(R / C).
        outputArray (np.ndarray): An N x R array in which the output features
            will be stored.
        chiArr (np.ndarray): A stack of diagonal matrices stored as an
            array of shape (R) drawn from a chi distribution.
        num_threads (int): This argument is so that this function has
            the same interface as the CPU SORF Transform. It is not
            needed for the GPU transform and is ignored.
        subtractMean (bool): If True, subtract the mean of each row from
            that row.
    """
    cdef const char *errCode
    cdef int i, startPosition, cutoff
    cdef np.ndarray[np.float32_t, ndim=3] reshapedXCopy
    cdef np.ndarray[np.float64_t, ndim=2] gradient
    cdef int num_repeats = (radem.shape[2] + reshapedX.shape[2] - 1) // reshapedX.shape[2]

    reshapedXCopy = reshapedX.copy()
    
    if reshapedX.shape[0] == 0:
        raise ValueError("There must be at least one datapoint.")
    if reshapedX.shape[0] != outputArray.shape[0]:
        raise ValueError("The number of input and output datapoints do not "
                "agree.")


    if radem.shape[0] != 3 or radem.shape[1] != 1:
        raise ValueError("radem must have length 3 for dim 0 and length 1 for dim1.")
    if outputArray.shape[1] != radem.shape[2]:
        raise ValueError("outputArray.shape[1] must be an integer multiple of the next largest "
                "power of 2 greater than the kernel width * X.shape[2].")

    if chiArr.shape[0] != radem.shape[2]:
        raise ValueError("chiArr.shape[0] must == radem.shape[2].")
        
    logdim = np.log2(reshapedX.shape[2])
    if np.ceil(logdim) != np.floor(logdim) or reshapedX.shape[2] < 2:
        raise ValueError("dim2 of the reshapedX array must be a power of 2 >= 2.")
    if not radem.shape[2] % reshapedX.shape[2] == 0:
        raise ValueError("The number of sampled frequencies should be an integer multiple of "
                "reshapedX.shape[2].")

    if not outputArray.flags["C_CONTIGUOUS"] or not reshapedX.flags["C_CONTIGUOUS"] \
            or not radem.flags["C_CONTIGUOUS"] or not chiArr.flags["C_CONTIGUOUS"]:
        raise ValueError("One or more arguments is not C contiguous.")

    startPosition, cutoff = 0, reshapedX.shape[2]
    
    for i in range(num_repeats):
        reshapedXCopy[:] = reshapedX
        errCode = floatConv1dPrep_(&radem[0,0,0],
                    &reshapedXCopy[0,0,0], numThreads,
                    reshapedX.shape[0], reshapedX.shape[1], 
                    reshapedX.shape[2], i * reshapedX.shape[2],
                    radem.shape[2])
        if errCode.decode("UTF-8") != "no_error":
            raise Exception("Fatal error encountered in floatGpuConv1dTransform_.")
        
        reshapedXCopy *= chiArr[None,None,(i * reshapedX.shape[2]):((i+1) * reshapedX.shape[2])]
        outputArray[:,startPosition:cutoff] = reshapedXCopy.max(axis=1)
        if subtractMean:
            outputArray[:,startPosition:cutoff] -= reshapedXCopy.mean(axis=1)

        cutoff += reshapedX.shape[2]
        startPosition += reshapedX.shape[2]
    


@cython.boundscheck(False)
@cython.wraparound(False)
def floatCpuConv1dFGen(np.ndarray[np.float32_t, ndim=3] reshapedX,
                np.ndarray[np.int8_t, ndim=3] radem,
                np.ndarray[np.float64_t, ndim=2] outputArray,
                np.ndarray[np.float32_t, ndim=1] chiArr,
                int numThreads, double beta_):
    """Uses wrapped C functions to generate random features for FHTConv1d, GraphConv1d,
    and related kernels. This function cannot be used to calculate the gradient
    so is only used for forward pass only (during fitting, inference, non-gradient-based
    optimization). It does not multiply by the lengthscales, so caller should do this.
    (This enables this function to also be used for GraphARD kernels if desired.)

    Args:
        reshapedX (np.ndarray): Raw data reshaped so that a convolution can be
            performed on it using orthogonal random features with the SORF
            operation. This array is not modified in place -- rather the features
            that are generated are stored in outputArray. Shape is (N x D x C) for 
            N datapoints. C must be a power of 2.
        radem (np.ndarray): A stack of diagonal matrices with elements drawn from the
            Rademacher distribution. Shape must be (3 x D x C).
        outputArray (np.ndarray): A numpy array in which the generated features will be
            stored. Is modified in-place.
        chiArr (np.ndarray): A stack of diagonal matrices stored as an
            array of shape m * C drawn from a chi distribution.
        num_threads (int): Number of threads to use for FHT.
        beta_ (float): The amplitude.

    Raises:
        ValueError: A ValueError is raised if unexpected or invalid inputs are supplied.

    Returns:
        gradient (np.ndarray); An array of shape output.shape[0] x output.shape[1] x 1.
    """
    cdef const char *errCode
    cdef np.ndarray[np.float32_t, ndim=3] reshapedXCopy = np.zeros((reshapedX.shape[0],
                        reshapedX.shape[1], reshapedX.shape[2]), dtype=np.float32)
    cdef double scalingTerm

    if reshapedX.shape[0] == 0:
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
        raise ValueError("radem should be an integer multiple of shape[2].")

    if not outputArray.flags["C_CONTIGUOUS"] or not reshapedX.flags["C_CONTIGUOUS"] or not radem.flags["C_CONTIGUOUS"] \
        or not chiArr.flags["C_CONTIGUOUS"]:
        raise ValueError("One or more arguments is not C contiguous.")

    scalingTerm = np.sqrt(1 / <double>chiArr.shape[0])
    scalingTerm *= beta_

    errCode = floatConvRBFFeatureGen_(&radem[0,0,0], &reshapedX[0,0,0],
                &reshapedXCopy[0,0,0], &chiArr[0], &outputArray[0,0],
                numThreads, reshapedX.shape[0],
                reshapedX.shape[1], reshapedX.shape[2],
                chiArr.shape[0], radem.shape[2])

    if errCode.decode("UTF-8") != "no_error":
        raise Exception("Fatal error encountered while performing convolution.")

    outputArray *= scalingTerm



@cython.boundscheck(False)
@cython.wraparound(False)
def floatCpuConvGrad(np.ndarray[np.float32_t, ndim=3] reshapedX,
                np.ndarray[np.int8_t, ndim=3] radem,
                np.ndarray[np.float64_t, ndim=2] outputArray,
                np.ndarray[np.float32_t, ndim=1] chiArr,
                int numThreads, float sigma,
                float beta_):
    """Performs feature generation for RBF-based convolution kernels while
    also performing gradient calculations.

    Args:
        reshapedX (np.ndarray): Raw data reshaped so that a convolution can be
            performed on it using orthogonal random features with the SORF
            operation. This array is not modified in place -- rather the features
            that are generated are stored in outputArray. Shape is (N x D x C) for 
            N datapoints. C must be a power of 2.
        radem (np.ndarray): A stack of diagonal matrices with elements drawn from the
            Rademacher distribution. Shape must be (3 x D x C).
        outputArray (np.ndarray): A numpy array in which the generated features will be
            stored. Is modified in-place.
        chiArr (np.ndarray): A stack of diagonal matrices stored as an
            array of shape m * C drawn from a chi distribution.
        num_threads (int): Number of threads to use for FHT.
        sigma (float): The lengthscale.
        beta_ (float): The amplitude.

    Raises:
        ValueError: A ValueError is raised if unexpected or invalid inputs are supplied.

    Returns:
        gradient (np.ndarray); An array of shape output.shape[0] x output.shape[1] x 1.
    """
    cdef const char *errCode
    cdef np.ndarray[np.float32_t, ndim=3] reshapedXCopy = np.zeros((reshapedX.shape[0], reshapedX.shape[1],
                        reshapedX.shape[2]), dtype=np.float32)
    cdef np.ndarray[np.float64_t, ndim=3] gradient = np.zeros((outputArray.shape[0], outputArray.shape[1], 1))
    cdef double scalingTerm

    if reshapedX.shape[0] == 0:
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

    if not outputArray.flags["C_CONTIGUOUS"] or not reshapedX.flags["C_CONTIGUOUS"] or not radem.flags["C_CONTIGUOUS"] \
        or not chiArr.flags["C_CONTIGUOUS"]:
        raise ValueError("One or more arguments to cpuGraphConv1dTransform is not "
                "C contiguous.")

    scalingTerm = np.sqrt(1 / <double>chiArr.shape[0])
    scalingTerm *= beta_

    errCode = floatConvRBFGrad_(&radem[0,0,0], &reshapedX[0,0,0],
                    &reshapedXCopy[0,0,0], &chiArr[0], &outputArray[0,0],
                    &gradient[0,0,0], sigma,
                    numThreads, reshapedX.shape[0],
                    reshapedX.shape[1], reshapedX.shape[2],
                    chiArr.shape[0], radem.shape[2])

    if errCode.decode("UTF-8") != "no_error":
        raise Exception("Fatal error encountered while performing graph convolution.")

    outputArray *= scalingTerm
    gradient *= scalingTerm
    return gradient


@cython.boundscheck(False)
@cython.wraparound(False)
def floatCpuGraphMiniARDGrad(np.ndarray[np.float32_t, ndim=3] inputX,
                np.ndarray[np.float64_t, ndim=2] randomFeats,
                np.ndarray[np.float32_t, ndim=2] precompWeights,
                np.ndarray[np.int32_t, ndim=1] sigmaMap,
                np.ndarray[np.float64_t, ndim=1] sigmaVals,
                double betaHparam, int numThreads):
    """Performs gradient calculations for the GraphMiniARD kernel, using
    pregenerated features and precomputed weights.

    Args:
        inputX (np.ndarray): The original input data.
        randomFeats (np.ndarray): The random features generated using the FHT-
            based procedure.
        precompWeights (np.ndarray): The FHT-rf gen procedure applied to an
            identity matrix.
        sigmaMap (np.ndarray): An array mapping which lengthscales correspond
            to which positions in the input.
        sigmaVals (np.ndarray): The lengthscale values, in an array of the same
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
    cdef np.ndarray[np.float64_t, ndim=3] gradient = np.zeros((randomFeats.shape[0],
                        randomFeats.shape[1], sigmaMap.max() + 1))

    if inputX.shape[0] == 0:
        raise ValueError("There must be at least one datapoint.")
    if inputX.shape[0] != randomFeats.shape[0] or precompWeights.shape[1] != inputX.shape[2]:
        raise ValueError("Incorrect array dims passed to a wrapped RBF "
                    "feature gen function.")
    if randomFeats.shape[1] != 2 * precompWeights.shape[0] or sigmaMap.shape[0] != \
            precompWeights.shape[1] or sigmaVals.shape[0] != sigmaMap.shape[0]:
        raise ValueError("Incorrect array dims passed to a wrapped RBF "
                    "feature gen function.")

    if not inputX.flags["C_CONTIGUOUS"] or not randomFeats.flags["C_CONTIGUOUS"] or not \
            precompWeights.flags["C_CONTIGUOUS"] or not sigmaMap.flags["C_CONTIGUOUS"] \
            or not sigmaVals.flags["C_CONTIGUOUS"]:
        raise ValueError("One or more arguments to a wrapped RBF feature gen function is not "
                "C contiguous.")

    rbfNormConstant = betaHparam * np.sqrt(1 / <double>precompWeights.shape[0])

    errCode = graphARDFloatGrad_(&inputX[0,0,0], &randomFeats[0,0],
                &precompWeights[0,0], &sigmaMap[0], &sigmaVals[0],
                &gradient[0,0,0], inputX.shape[0], inputX.shape[1],
                inputX.shape[2],
                gradient.shape[2], precompWeights.shape[0],
                rbfNormConstant, numThreads)
    if errCode.decode("UTF-8") != "no_error":
        raise Exception("Fatal error encountered in RBF feature gen.")
    return gradient
