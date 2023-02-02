"""Wraps the C functions that perform the fastHadamardTransform on CPU,
the structured orthogonal features or SORF operations on CPU and the
fast Hadamard transform based convolution operations on CPU.
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
    const char *doubleConv1dPrep_(int8_t *radem, double *reshapedX,
            int numThreads, int reshapedDim0,
            int reshapedDim1, int reshapedDim2,
            int startPosition, int numFreqs);

cdef extern from "transform_functions.h" nogil:
    const char *SORFDoubleBlockTransform_(double *Z, int8_t *radem, int zDim0,
        int zDim1, int zDim2, int numThreads)


@cython.boundscheck(False)
@cython.wraparound(False)
def doubleCpuConv1dTransform(np.ndarray[np.float64_t, ndim=3] reshapedX,
                np.ndarray[np.int8_t, ndim=3] radem,
                np.ndarray[np.float64_t, ndim=2] Z,
                np.ndarray[np.float64_t, ndim=1] S,
                int numThreads, double sigma, 
                str mode):
    """Uses the wrapped doubleConv1dPrep_ and numpy operations to perform a
    structured orthogonal random features or SORF operation on
    an array of doubles that has been reshaped to perform convolution.
    Note that doubleConv1dPrep should ONLY be accessed through this
    since this wrapper performs key checks (the shape of the input
    arrays, are they C-contiguous etc.) that should not be bypassed.

    Args:
        reshapedX (np.ndarray): An array of type float64 from which
            the features will be generated. Is not modified. Must
            be of shape (N x D x C) where C is a power of 2. Should
            have been reshaped to be appropriate for convolution.
        radem (np.ndarray): A stack of diagonal matrices of type int8_t
            of shape (3 x 1 x m * C), where R is the number of random
            features requested and m is ceil(R / C).
        Z (np.ndarray): An N x R array in which the output features
            will be stored.
        S (np.ndarray): A stack of diagonal matrices stored as an
            array of shape (R) drawn from a chi distribution.
        num_threads (int): This argument is so that this function has
            the same interface as the CPU SORF Transform. It is not
            needed for the GPU transform and is ignored.
        sigma (double): The kernel specific hyperparameter for the
            Conv1d kernel.
        mode (str): One of 'maxpool', 'maxpool_loc', 'conv', 'conv_gradient'.
            Determines the type of activation function and pooling that
            is performed.

    Returns:
        gradient (np.ndarray): A float64 array of shape (N x R) if
            mode is "conv_gradient"; otherwise nothing.
    """
    cdef const char *errCode
    cdef int i, startPosition, cutoff
    cdef double scalingTerm
    cdef np.ndarray[np.float64_t, ndim=3] reshapedXCopy
    cdef np.ndarray[np.float64_t, ndim=2] gradient
    cdef int num_repeats = (radem.shape[2] + reshapedX.shape[2] - 1) // reshapedX.shape[2]

    reshapedXCopy = np.zeros((reshapedX.shape[0], reshapedX.shape[1], reshapedX.shape[2]))
    
    if mode not in ["maxpool", "maxpool_loc", "conv", "conv_gradient"]:
        raise ValueError("Invalid mode supplied for convolution.")

    #Check that all arrays have expected sizes and data types.    
    if reshapedX.shape[0] == 0:
        raise ValueError("There must be at least one datapoint.")
    if reshapedX.shape[0] != Z.shape[0]:
        raise ValueError("The number of input and output datapoints do not "
                "agree.")


    #Check that shapes of radem, Z are correct.
    if radem.shape[0] != 3 or radem.shape[1] != 1:
        raise ValueError("radem must have length 3 for dim 0 and length 1 for dim1.")
    if mode.startswith("maxpool"):
        if Z.shape[1] != radem.shape[2]:
            raise ValueError("Z.shape[1] must be an integer multiple of the next largest "
                "power of 2 greater than the kernel width * X.shape[2].")
    elif Z.shape[1] != 2 * radem.shape[2]:
        raise ValueError("Z.shape[1] must be 2 * radem.shape[2], which must be an integer "
                "multiple of the next largest power of 2 greater than the kernel width * "
                "X.shape[2].")

    #Next, make sure that reshapedX and S make sense.
    if S.shape[0] != radem.shape[2]:
        raise ValueError("S.shape[0] must == radem.shape[2].")
        
    logdim = np.log2(reshapedX.shape[2])
    if np.ceil(logdim) != np.floor(logdim) or reshapedX.shape[2] < 2:
        raise ValueError("dim2 of the reshapedX array must be a power of 2 >= 2.")
    if not radem.shape[2] % reshapedX.shape[2] == 0:
        raise ValueError("The number of sampled frequencies should be an integer multiple of "
                "reshapedX.shape[2].")

    #Make sure that all inputs are C-contiguous.
    if not Z.flags["C_CONTIGUOUS"] or not reshapedX.flags["C_CONTIGUOUS"] \
            or not radem.flags["C_CONTIGUOUS"] or not S.flags["C_CONTIGUOUS"]:
        raise ValueError("One or more arguments is not C contiguous.")

    startPosition, cutoff = 0, reshapedX.shape[2]
    scalingTerm = np.sqrt(1 / <double>radem.shape[2])
    if mode == "conv_gradient":
        gradient = np.zeros((Z.shape[0], Z.shape[1]), dtype=np.float64)
    
    for i in range(num_repeats):
        reshapedXCopy[:] = reshapedX
        errCode = doubleConv1dPrep_(&radem[0,0,0],
                    &reshapedXCopy[0,0,0], numThreads,
                    reshapedX.shape[0], reshapedX.shape[1], 
                    reshapedX.shape[2], i * reshapedX.shape[2],
                    radem.shape[2])
        if errCode.decode("UTF-8") != "no_error":
            raise Exception("Fatal error encountered in doubleGpuConv1dTransform_.")
        
        reshapedXCopy *= S[None,None,(i * reshapedX.shape[2]):((i+1) * reshapedX.shape[2])]
        if mode == "maxpool":
            Z[:,startPosition:cutoff] = reshapedXCopy.max(axis=1)
        elif mode == "maxpool_loc":
            Z[:,startPosition:cutoff] = reshapedXCopy.max(axis=1)
            Z[:,startPosition:cutoff] -= reshapedXCopy.mean(axis=1)
        elif mode == "conv_gradient":
            featureMod = sigma * reshapedXCopy

            Z[:,startPosition:cutoff] = np.cos(featureMod).sum(axis=1)
            gradient[:,startPosition:cutoff] = (-np.sin(featureMod) * \
                                reshapedXCopy).sum(axis=1)
            cutoff += reshapedX.shape[2]
            startPosition += reshapedX.shape[2]
            Z[:,startPosition:cutoff] = np.sin(featureMod).sum(axis=1)
            gradient[:,startPosition:cutoff] = (np.cos(featureMod) * \
                                reshapedXCopy).sum(axis=1)
        elif mode == "conv":
            reshapedXCopy *= sigma
            Z[:,startPosition:cutoff] = np.sum(np.cos(reshapedXCopy), axis=1)
            cutoff += reshapedX.shape[2]
            startPosition += reshapedX.shape[2]
            Z[:,startPosition:cutoff] = np.sum(np.sin(reshapedXCopy), axis=1)

        cutoff += reshapedX.shape[2]
        startPosition += reshapedX.shape[2]
    
    if not mode.startswith("maxpool"):
        Z *= scalingTerm
    if mode == "conv_gradient":
        gradient *= scalingTerm
        return gradient



@cython.boundscheck(False)
@cython.wraparound(False)
def doubleCpuGraphConv1dTransform(np.ndarray[np.float64_t, ndim=3] reshapedX,
                np.ndarray[np.int8_t, ndim=3] radem,
                np.ndarray[np.float64_t, ndim=2] Z,
                np.ndarray[np.float64_t, ndim=1] S,
                int numThreads, double sigma,
                double beta_,
                str mode = "standard"):
    """Uses the wrapped doubleConv1dPrep_ and numpy operations to perform
    1d convolution on a sequence of vectors representing the nodes
    of a graph.

    Args:
        reshapedX (np.ndarray): Raw data reshaped so that a convolution can be
            performed on it using orthogonal random features with the SORF
            operation. This array is not modified in place -- rather the features
            that are generated are stored in Z. Shape is (N x D x C) for 
            N datapoints. C must be a power of 2.
        radem (np.ndarray): A stack of diagonal matrices with elements drawn from the
            Rademacher distribution. Shape must be (3 x D x C).
        Z (np.ndarray): A numpy array in which the generated features will be
            stored. Is modified in-place.
        S (np.ndarray): A stack of diagonal matrices stored as an
            array of shape m * C drawn from a chi distribution.
        num_threads (int): Number of threads to use for FHT.
        sigma (double): The lengthscale.
        beta_ (double): The amplitude.
        mode (str): Either 'standard' or 'gradient'. If 'gradient' a gradient array is returned.

    Raises:
        ValueError: A ValueError is raised if unexpected or invalid inputs are supplied.

    Returns:
        gradient (np.ndarray); An array of shape output.shape[0] x output.shape[1] x 1.
            Only returned if mode == "gradient"; otherwise, nothing is returned.
    """
    cdef const char *errCode
    cdef np.ndarray[np.float64_t, ndim=3] reshapedXCopy = reshapedX.copy()
    cdef np.ndarray[np.float64_t, ndim=3] featureMod
    cdef np.ndarray[np.float64_t, ndim=3] gradient
    cdef float scalingTerm
    cdef int num_repeats = (radem.shape[2] + reshapedX.shape[2] - 1) // reshapedX.shape[2]
    cdef int startPosition, cutoff, startPos2, cutoff2, i, j

    if mode not in ["standard", "gradient"]:
        raise ValueError("Invalid mode supplied for convolution.")

    if reshapedX.shape[0] == 0:
        raise ValueError("There must be at least one datapoint.")
    if reshapedX.shape[0] != Z.shape[0]:
        raise ValueError("The number of datapoints in the outputs and the inputs do "
                "not agree.")
    if radem.shape[0] != 3 or radem.shape[1] != 1:
        raise ValueError("radem must have length 3 for dim 0 and length 1 for dim1.")
    if Z.shape[1] != 2 * radem.shape[2]:
        raise ValueError("Z.shape[1] must be 2 * radem.shape[2], which must be an integer multiple of "
                    "the next power of 2 greater than the kernel width * X.shape[2].")
    
    if S.shape[0] != radem.shape[2]:
        raise ValueError("S.shape[0] must == radem.shape[2].")
    logdim = np.log2(reshapedX.shape[2])
    if np.ceil(logdim) != np.floor(logdim) or reshapedX.shape[2] < 2:
        raise ValueError("dim2 of the reshapedX array must be a power of 2 >= 2.")
    if not radem.shape[2] % reshapedX.shape[2] == 0:
        raise ValueError("The number of sampled frequencies should be an integer multiple of "
                "reshapedX.shape[2].")

    if not Z.flags["C_CONTIGUOUS"] or not reshapedX.flags["C_CONTIGUOUS"] or not radem.flags["C_CONTIGUOUS"] \
        or not S.flags["C_CONTIGUOUS"]:
        raise ValueError("One or more arguments to cpuGraphConv1dTransform is not "
                "C contiguous.")

    featureMod = np.zeros((reshapedX.shape[0], reshapedX.shape[1], reshapedX.shape[2]))
    startPosition, cutoff = 0, reshapedX.shape[2]
    startPos2, cutoff2 = reshapedX.shape[2], cutoff + reshapedX.shape[2]
    scalingTerm = np.sqrt(1 / <double>radem.shape[2])
    scalingTerm *= beta_

    if mode == "gradient":
        gradient = np.zeros((Z.shape[0], Z.shape[1], 1))

    for i in range(num_repeats):
        reshapedXCopy[:] = reshapedX * sigma
        errCode = doubleConv1dPrep_(&radem[0,0,0],
                    &reshapedXCopy[0,0,0], numThreads,
                    reshapedX.shape[0], reshapedX.shape[1], 
                    reshapedX.shape[2], i * reshapedX.shape[2],
                    radem.shape[2])

        if errCode.decode("UTF-8") != "no_error":
            raise Exception("Fatal error encountered while performing graph convolution.")
        reshapedXCopy *= S[None,None,(i * reshapedX.shape[2]):((i+1) * reshapedX.shape[2])]

        if mode == "gradient":
            gradient[:,startPosition:cutoff,0] = (-np.sin(reshapedXCopy) * reshapedXCopy /
                                            sigma).sum(axis=1)
            gradient[:,startPos2:cutoff2,0] = (np.cos(reshapedXCopy) * reshapedXCopy /
                                        sigma).sum(axis=1)

        Z[:,startPosition:cutoff] = np.sum(np.cos(reshapedXCopy), axis=1)
        Z[:,startPos2:cutoff2] = np.sum(np.sin(reshapedXCopy), axis=1)
        cutoff += 2 * reshapedX.shape[2]
        startPosition += 2 * reshapedX.shape[2]
        startPos2 += 2 * reshapedX.shape[2]
        cutoff2 += 2 * reshapedX.shape[2]

    Z *= scalingTerm
    if mode == "gradient":
        gradient *= scalingTerm
        return gradient




@cython.boundscheck(False)
@cython.wraparound(False)
def doubleCpuGraphPolyFHT(np.ndarray[np.float64_t, ndim=3] reshapedX,
                np.ndarray[np.int8_t, ndim=3] radem,
                np.ndarray[np.float64_t, ndim=2] S,
                np.ndarray[np.float64_t, ndim=2] Z,
                int polydegree, int numThreads):
    """Uses the wrapped PolyFHT_ and numpy operations to apply a pairwise
    polynomial kernel to all elements of two graphs.

    Args:
        reshapedX (np.ndarray): Raw data reshaped so that the random features
            transformation can be applied. This array is not modified in place --
            rather the features that are generated are stored in Z. Shape is (N x D x C)
            for N datapoints. C must be a power of 2.
        radem (np.ndarray): A stack of diagonal matrices with elements drawn from the
            Rademacher distribution. Shape must be (polydegree x 1 x C).
        S (np.ndarray): A stack of diagonal matrices stored as an
            array of shape (polydegree, m * C) drawn from a chi distribution.
        Z (np.ndarray): A numpy array in which the generated features will be
            stored. Is modified in-place.
        polydegree (int): The degree of the polynomial kernel that we approximate. Should
            be <= 4 (for very high-degree polynomial kernels we are probably better off
            switching to an RBF or convolution kernel).
        num_threads (int): Number of threads to use for FHT.

    Raises:
        ValueError: A ValueError is raised if unexpected or invalid inputs are supplied.
    """
    cdef const char *errCode
    cdef np.ndarray[np.float64_t, ndim=3] reshapedXCopy = reshapedX.copy()
    cdef np.ndarray[np.float64_t, ndim=3] preSumFeats = reshapedX.copy()
    cdef float scalingTerm
    cdef int num_repeats = (radem.shape[2] + reshapedX.shape[2] - 1) // reshapedX.shape[2]
    cdef int startPosition, cutoff, i, j
    
    

    if reshapedX.shape[0] == 0:
        raise ValueError("There must be at least one datapoint.")
    if reshapedX.shape[0] != Z.shape[0]:
        raise ValueError("The number of datapoints in the outputs and the inputs do "
                "not agree.")
    if radem.shape[0] != 3 * polydegree or radem.shape[1] != 1:
        raise ValueError("radem must have length polydegree for dim 0 and length 1 for dim1.")
    if Z.shape[1] != radem.shape[2]:
        raise ValueError("Z.shape[1] must be radem.shape[2], which must be an integer multiple of "
                    "the next power of 2 greater than the kernel width * X.shape[2].")
    
    if S.shape[1] != radem.shape[2]:
        raise ValueError("S.shape[1] must == radem.shape[2].")
    if S.shape[0] != polydegree:
        raise ValueError("S.shape[0] must == polydegree.")
    logdim = np.log2(reshapedX.shape[2])
    if np.ceil(logdim) != np.floor(logdim) or reshapedX.shape[2] < 2:
        raise ValueError("dim2 of the reshapedX array must be a power of 2 >= 2.")
    if not radem.shape[2] % reshapedX.shape[2] == 0:
        raise ValueError("The number of sampled frequencies should be an integer multiple of "
                "reshapedX.shape[2].")

    if not Z.flags["C_CONTIGUOUS"] or not reshapedX.flags["C_CONTIGUOUS"] or not radem.flags["C_CONTIGUOUS"] \
        or not S.flags["C_CONTIGUOUS"]:
        raise ValueError("One or more arguments is not C contiguous.")

    startPosition, cutoff = 0, reshapedX.shape[2]
    scalingTerm = np.sqrt(1 / <double>radem.shape[2])

    for i in range(num_repeats):
        preSumFeats[:] = reshapedX
        errCode = doubleConv1dPrep_(&radem[0,0,0],
                    &preSumFeats[0,0,0], numThreads,
                    reshapedX.shape[0], reshapedX.shape[1], 
                    reshapedX.shape[2], i * reshapedX.shape[2],
                    radem.shape[2])

        if errCode.decode("UTF-8") != "no_error":
            raise Exception("Fatal error encountered while performing FHT RF generation.")
        preSumFeats *= S[0:1, None, startPosition:cutoff]
        for j in range(1, polydegree):
            reshapedXCopy[:] = reshapedX
            errCode = doubleConv1dPrep_(&radem[3*j,0,0],
                    &reshapedXCopy[0,0,0], numThreads,
                    reshapedX.shape[0], reshapedX.shape[1], 
                    reshapedX.shape[2], i * reshapedX.shape[2],
                    radem.shape[2])
            reshapedXCopy *= S[j:(j+1), None, startPosition:cutoff]
            preSumFeats *= reshapedXCopy

        Z[:,startPosition:cutoff] = np.sum(preSumFeats, axis=1)

        cutoff += reshapedX.shape[2]
        startPosition += reshapedX.shape[2]

    Z *= scalingTerm



@cython.boundscheck(False)
@cython.wraparound(False)
def doubleCpuPolyFHT(np.ndarray[np.float64_t, ndim=3] reshapedX,
                np.ndarray[np.int8_t, ndim=3] radem,
                np.ndarray[np.float64_t, ndim=3] S,
                np.ndarray[np.float64_t, ndim=3] Z,
                int polydegree, int numThreads):
    """Uses the wrapped PolyFHT_ and numpy operations for a polynomial
    kernel on fixed vector data in a 3d array.

    Args:
        reshapedX (np.ndarray): Raw data reshaped so that the random features
            transformation can be applied. This array is not modified in place --
            rather the features that are generated are stored in Z. Shape is (N x D x C)
            for N datapoints. C must be a power of 2.
        radem (np.ndarray): A stack of diagonal matrices with elements drawn from the
            Rademacher distribution. Shape must be (polydegree x D x C).
        S (np.ndarray): A stack of diagonal matrices stored as an
            array of shape (polydegree, D, C) drawn from a chi distribution.
        Z (np.ndarray): A numpy array in which the generated features will be
            stored. Is modified in-place.
        polydegree (int): The degree of the polynomial kernel that we approximate. Should
            be <= 4 (for very high-degree polynomial kernels we are probably better off
            switching to an RBF or convolution kernel).
        num_threads (int): Number of threads to use for FHT.

    Raises:
        ValueError: A ValueError is raised if unexpected or invalid inputs are supplied.
    """
    cdef const char *errCode
    cdef np.ndarray[np.float64_t, ndim=3] reshapedXCopy = reshapedX.copy()
    cdef float scalingTerm
    cdef int j
    
    

    if reshapedX.shape[0] == 0:
        raise ValueError("There must be at least one datapoint.")
    if reshapedX.shape[0] != Z.shape[0] or reshapedX.shape[1] != Z.shape[1] or\
            reshapedX.shape[2] != Z.shape[2]:
        raise ValueError("The number of datapoints in the outputs and the inputs do "
                "not agree.")
    if radem.shape[0] != 3 * polydegree or S.shape[0] != polydegree:
        raise ValueError("radem & S must have length polydegree for dim 0.")
    
    if S.shape[2] != radem.shape[2] or S.shape[1] != radem.shape[1]:
        raise ValueError("S must have same shape[1] and shape[2] as radem.")
    logdim = np.log2(reshapedX.shape[2])
    if np.ceil(logdim) != np.floor(logdim) or reshapedX.shape[2] < 2:
        raise ValueError("dim2 of the reshapedX array must be a power of 2 >= 2.")
    if radem.shape[2] != reshapedX.shape[2] or radem.shape[1] != reshapedX.shape[1]:
        raise ValueError("reshapedX shape[1] and shape[2] must == radem shape[1] and shape[2].")

    if not Z.flags["C_CONTIGUOUS"] or not reshapedX.flags["C_CONTIGUOUS"] or not radem.flags["C_CONTIGUOUS"] \
        or not S.flags["C_CONTIGUOUS"]:
        raise ValueError("One or more arguments is not C contiguous.")

    scalingTerm = np.sqrt(1 / <double>radem.shape[2])

    Z[:] = reshapedX
    errCode = SORFDoubleBlockTransform_(&Z[0,0,0], &radem[0,0,0],
                        reshapedX.shape[0], reshapedX.shape[1], reshapedX.shape[2],
                        numThreads)
    Z *= S[0:1, :, :]

    if errCode.decode("UTF-8") != "no_error":
        raise Exception("Fatal error encountered while performing graph convolution.")
    

    for j in range(1, polydegree):
        reshapedXCopy[:] = reshapedX
        errCode = SORFDoubleBlockTransform_(&reshapedXCopy[0,0,0], &radem[3*j,0,0],
                        reshapedX.shape[0], reshapedX.shape[1], reshapedX.shape[2],
                        numThreads)
        reshapedXCopy *= S[j:j+1, :, :]
        Z *= reshapedXCopy

    Z *= scalingTerm
