"""Wraps the C functions that perform the fastHadamardTransform on CPU,
the structured orthogonal features or SORF operations on CPU and the
fast Hadamard transform based convolution operations on CPU for float arrays.
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



cdef extern from "conv1d_operations.h" nogil:
    const char *floatConv1dPrep_(int8_t *radem, float *reshapedX,
            int numThreads, int reshapedDim0,
            int reshapedDim1, int reshapedDim2,
            int startPosition, int numFreqs)
    const char *floatConvRBFFeatureGen_(int8_t *radem, float *reshapedX,
            float *chiArr, double *outputArray,
            int numThreads, int reshapedDim0,
            int reshapedDim1, int reshapedDim2,
            int startPosition, int numFreqs)

cdef extern from "transform_functions.h" nogil:
    const char *SORFFloatBlockTransform_(float *Z, int8_t *radem, int zDim0,
            int zDim1, int zDim2, int numThreads)


@cython.boundscheck(False)
@cython.wraparound(False)
def floatCpuConv1dMaxpool(np.ndarray[np.float32_t, ndim=3] reshapedX,
                np.ndarray[np.int8_t, ndim=3] radem,
                np.ndarray[np.float64_t, ndim=2] outputArray,
                np.ndarray[np.float32_t, ndim=1] chiArr,
                int numThreads, str mode):
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
        mode (str): One of 'maxpool', 'maxpool_loc'.
            Determines the type of activation function and pooling that
            is performed.

    Returns:
        gradient (np.ndarray): A float32 array of shape (N x R) if
            mode is "conv_gradient"; otherwise nothing.
    """
    cdef const char *errCode
    cdef int i, startPosition, cutoff
    cdef float scalingTerm
    cdef np.ndarray[np.float32_t, ndim=3] reshapedXCopy
    cdef np.ndarray[np.float64_t, ndim=2] gradient
    cdef int num_repeats = (radem.shape[2] + reshapedX.shape[2] - 1) // reshapedX.shape[2]

    reshapedXCopy = reshapedX.copy()
    
    if mode not in ["maxpool", "maxpool_loc"]:
        raise ValueError("Invalid mode supplied for convolution.")

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
    scalingTerm = np.sqrt(1 / <float>radem.shape[2])
    
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
        if mode == "maxpool_loc":
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
            Only returned if mode == "gradient"; otherwise, nothing is returned.
    """
    cdef const char *errCode
    cdef np.ndarray[np.float32_t, ndim=3] reshapedXCopy = np.zeros((reshapedX.shape[0],
                        reshapedX.shape[1], reshapedX.shape[2]))
    cdef double scalingTerm
    cdef int num_repeats = (radem.shape[2] + reshapedX.shape[2] - 1) // reshapedX.shape[2]
    cdef int i

    if reshapedX.shape[0] == 0:
        raise ValueError("There must be at least one datapoint.")
    if reshapedX.shape[0] != outputArray.shape[0]:
        raise ValueError("The number of datapoints in the outputs and the inputs do "
                "not agree.")
    if radem.shape[0] != 3 or radem.shape[1] != 1:
        raise ValueError("radem must have length 3 for dim 0 and length 1 for dim1.")
    if outputArray.shape[1] != 2 * radem.shape[2]:
        raise ValueError("outputArray.shape[1] must be 2 * radem.shape[2], which must be an integer multiple of "
                    "the next power of 2 greater than the kernel width * X.shape[2].")
    
    if chiArr.shape[0] != radem.shape[2]:
        raise ValueError("chiArr.shape[0] must == radem.shape[2].")
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

    scalingTerm = np.sqrt(1 / <double>radem.shape[2])
    scalingTerm *= beta_

    for i in range(num_repeats):
        errCode = floatConvRBFFeatureGen_(&radem[0,0,0], &reshapedX[0,0,0],
                &reshapedXCopy[0,0,0], &chiArr[0], &outputArray[0,0], numThreads, reshapedX.shape[0],
                reshapedX.shape[1], reshapedX.shape[2], i * reshapedX.shape[2], radem.shape[2])

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
    cdef np.ndarray[np.float32_t, ndim=3] reshapedXCopy = reshapedX.copy()
    cdef np.ndarray[np.float32_t, ndim=3] featureMod
    cdef np.ndarray[np.float64_t, ndim=3] gradient
    cdef float scalingTerm
    cdef int num_repeats = (radem.shape[2] + reshapedX.shape[2] - 1) // reshapedX.shape[2]
    cdef int startPosition, cutoff, startPos2, cutoff2, i, j

    if reshapedX.shape[0] == 0:
        raise ValueError("There must be at least one datapoint.")
    if reshapedX.shape[0] != outputArray.shape[0]:
        raise ValueError("The number of datapoints in the outputs and the inputs do "
                "not agree.")
    if radem.shape[0] != 3 or radem.shape[1] != 1:
        raise ValueError("radem must have length 3 for dim 0 and length 1 for dim1.")
    if outputArray.shape[1] != 2 * radem.shape[2]:
        raise ValueError("outputArray.shape[1] must be 2 * radem.shape[2], which must be an integer multiple of "
                    "the next power of 2 greater than the kernel width * X.shape[2].")
    
    if chiArr.shape[0] != radem.shape[2]:
        raise ValueError("chiArr.shape[0] must == radem.shape[2].")
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

    featureMod = np.zeros((reshapedX.shape[0], reshapedX.shape[1], reshapedX.shape[2]), dtype=np.float32)
    startPosition, cutoff = 0, reshapedX.shape[2]
    startPos2, cutoff2 = reshapedX.shape[2], cutoff + reshapedX.shape[2]
    scalingTerm = np.sqrt(1 / <float>radem.shape[2])
    scalingTerm *= beta_

    gradient = np.zeros((outputArray.shape[0], outputArray.shape[1], 1))

    for i in range(num_repeats):
        reshapedXCopy[:] = reshapedX * sigma

        errCode = floatConv1dPrep_(&radem[0,0,0],
                    &reshapedXCopy[0,0,0], numThreads,
                    reshapedX.shape[0], reshapedX.shape[1], 
                    reshapedX.shape[2], i * reshapedX.shape[2],
                    radem.shape[2])

        if errCode.decode("UTF-8") != "no_error":
            raise Exception("Fatal error encountered while performing graph convolution.")
        reshapedXCopy *= chiArr[None,None,(i * reshapedX.shape[2]):((i+1) * reshapedX.shape[2])]

        gradient[:,startPosition:cutoff,0] = (-np.sin(reshapedXCopy) * reshapedXCopy /
                                            sigma).sum(axis=1)
        gradient[:,startPos2:cutoff2,0] = (np.cos(reshapedXCopy) * reshapedXCopy /
                                        sigma).sum(axis=1)

        outputArray[:,startPosition:cutoff] = np.sum(np.cos(reshapedXCopy), axis=1)
        outputArray[:,startPos2:cutoff2] = np.sum(np.sin(reshapedXCopy), axis=1)
        cutoff += 2 * reshapedX.shape[2]
        startPosition += 2 * reshapedX.shape[2]
        startPos2 += 2 * reshapedX.shape[2]
        cutoff2 += 2 * reshapedX.shape[2]

    outputArray *= scalingTerm
    gradient *= scalingTerm
    return gradient



@cython.boundscheck(False)
@cython.wraparound(False)
def floatCpuGraphPolyFHT(np.ndarray[np.float32_t, ndim=3] reshapedX,
                np.ndarray[np.int8_t, ndim=3] radem,
                np.ndarray[np.float32_t, ndim=2] chiArr,
                np.ndarray[np.float32_t, ndim=2] outputArray,
                int polydegree, int numThreads):
    """Uses the wrapped PolyFHT_ and numpy operations to apply a pairwise
    polynomial kernel to all elements of two graphs.

    Args:
        reshapedX (np.ndarray): Raw data reshaped so that the random features
            transformation can be applied. This array is not modified in place --
            rather the features that are generated are stored in outputArray. Shape is (N x D x C)
            for N datapoints. C must be a power of 2.
        radem (np.ndarray): A stack of diagonal matrices with elements drawn from the
            Rademacher distribution. Shape must be (polydegree x 1 x C).
        chiArr (np.ndarray): A stack of diagonal matrices stored as an
            array of shape (polydegree, m * C) drawn from a chi distribution.
        outputArray (np.ndarray): A numpy array in which the generated features will be
            stored. Is modified in-place.
        polydegree (int): The degree of the polynomial kernel that we approximate. Should
            be <= 4 (for very high-degree polynomial kernels we are probably better off
            switching to an RBF or convolution kernel).
        num_threads (int): Number of threads to use for FHT.

    Raises:
        ValueError: A ValueError is raised if unexpected or invalid inputs are supplied.
    """
    cdef const char *errCode
    cdef np.ndarray[np.float32_t, ndim=3] reshapedXCopy = reshapedX.copy()
    cdef np.ndarray[np.float32_t, ndim=3] preSumFeats = reshapedX.copy()
    cdef float scalingTerm
    cdef int num_repeats = (radem.shape[2] + reshapedX.shape[2] - 1) // reshapedX.shape[2]
    cdef int startPosition, cutoff, i, j
    
    

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

    startPosition, cutoff = 0, reshapedX.shape[2]
    scalingTerm = np.sqrt(1 / <float>radem.shape[2])

    for i in range(num_repeats):
        preSumFeats[:] = reshapedX
        errCode = floatConv1dPrep_(&radem[0,0,0],
                    &preSumFeats[0,0,0], numThreads,
                    reshapedX.shape[0], reshapedX.shape[1], 
                    reshapedX.shape[2], i * reshapedX.shape[2],
                    radem.shape[2])

        if errCode.decode("UTF-8") != "no_error":
            raise Exception("Fatal error encountered while performing FHT RF generation.")
        preSumFeats *= chiArr[0:1, None, startPosition:cutoff]
        for j in range(1, polydegree):
            reshapedXCopy[:] = reshapedX
            errCode = floatConv1dPrep_(&radem[3*j,0,0],
                    &reshapedXCopy[0,0,0], numThreads,
                    reshapedX.shape[0], reshapedX.shape[1], 
                    reshapedX.shape[2], i * reshapedX.shape[2],
                    radem.shape[2])
            reshapedXCopy *= chiArr[j:(j+1), None, startPosition:cutoff]
            preSumFeats *= reshapedXCopy

        outputArray[:,startPosition:cutoff] = np.sum(preSumFeats, axis=1)

        cutoff += reshapedX.shape[2]
        startPosition += reshapedX.shape[2]

    outputArray *= scalingTerm



@cython.boundscheck(False)
@cython.wraparound(False)
def floatCpuPolyFHT(np.ndarray[np.float32_t, ndim=3] reshapedX,
                np.ndarray[np.int8_t, ndim=3] radem,
                np.ndarray[np.float32_t, ndim=3] chiArr,
                np.ndarray[np.float32_t, ndim=3] outputArray,
                int polydegree, int numThreads):
    """Uses the wrapped PolyFHT_ and numpy operations to apply a pairwise
    polynomial kernel to all elements of two graphs.

    Args:
        reshapedX (np.ndarray): Raw data reshaped so that the random features
            transformation can be applied. This array is not modified in place --
            rather the features that are generated are stored in outputArray. Shape is (N x D x C)
            for N datapoints. C must be a power of 2.
        radem (np.ndarray): A stack of diagonal matrices with elements drawn from the
            Rademacher distribution. Shape must be (polydegree x D x C).
        chiArr (np.ndarray): A stack of diagonal matrices stored as an
            array of shape (polydegree, D, C) drawn from a chi distribution.
        permutator (cp.ndarray): Array of shape (polydegree x D x C) that permutes the
            columns of reshapedX when random features are generated.
        outputArray (np.ndarray): A numpy array in which the generated features will be
            stored. Is modified in-place.
        polydegree (int): The degree of the polynomial kernel that we approximate. Should
            be <= 4 (for very high-degree polynomial kernels we are probably better off
            switching to an RBF or convolution kernel).
        num_threads (int): Number of threads to use for FHT.

    Raises:
        ValueError: A ValueError is raised if unexpected or invalid inputs are supplied.
    """
    cdef const char *errCode
    cdef np.ndarray[np.float32_t, ndim=3] reshapedXCopy = reshapedX.copy()
    cdef float scalingTerm
    cdef int j
    
    

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


    scalingTerm = np.sqrt(1 / <float>radem.shape[2])

    outputArray[:] = reshapedX
    errCode = SORFFloatBlockTransform_(&outputArray[0,0,0], &radem[0,0,0],
                        reshapedX.shape[0], reshapedX.shape[1], reshapedX.shape[2],
                        numThreads)
    outputArray *= chiArr[0:1, :, :]

    if errCode.decode("UTF-8") != "no_error":
        raise Exception("Fatal error encountered while performing graph convolution.")

    for j in range(1, polydegree):
        reshapedXCopy[:] = reshapedX
        errCode = SORFFloatBlockTransform_(&reshapedXCopy[0,0,0], &radem[3*j,0,0],
                        reshapedX.shape[0], reshapedX.shape[1], reshapedX.shape[2],
                        numThreads)
        reshapedXCopy *= chiArr[j:j+1, :, :]
        outputArray *= reshapedXCopy

    outputArray *= scalingTerm
