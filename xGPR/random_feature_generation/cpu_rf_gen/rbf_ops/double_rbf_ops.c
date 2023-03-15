/*!
 * # double_rbf_ops.c
 *
 * This module performs all major steps involved in feature generation for
 * RBF-type kernels, which includes RBF, Matern, ARD and MiniARD (and by extension
 * the static layer kernels). Functions from array_operations are used to perform
 * the fast Hadamard transform and rademacher matrix multiplication pieces.
 * The "specialized" piece, multiplication by a diagonal matrix while performing
 * sine-cosine operations, is performed here.
 *
 * + rbfFeatureGenDouble_
 * Performs the feature generation steps on an input array of doubles.
 *
 * + rbfDoubleGrad_
 * Performs the feature generation steps on an input array of doubles
 * AND generates the gradient info (stored in a separate array). For non-ARD
 * kernels only.
 *
 * + ardDoubleGrad_
 * Performs gradient and feature generation calculations for an RBF ARD kernel.
 * Slower than rbfFeatureGen, so use only if gradient is required.
 *
 * + ThreadRBFGenDouble
 * Performs operations for a single thread of the feature generation operation.
 *
 * + ThreadRBFDoubleGrad
 * Performs operations for a single thread of the gradient / feature operation.
 * 
 * + ThreadARDDoubleGrad
 * Performs operations for a single thread of the ARD gradient-only calculation.
 *
 * + rbfDoubleFeatureGenLastStep_
 * Performs the final operations involved in the feature generation for doubles.
 *
 * + rbfDoubleGradLastStep_
 * Performs the final operations involved in feature / gradient calc for doubles.
 *
 * + ardDoubleGradCalcs
 * Performs the key operations involved in gradient-only calc for ARD.
 */
#include <Python.h>
#include <pthread.h>
#include <stdint.h>
#include <math.h>
#include "double_rbf_ops.h"
#include "../shared_fht_functions/float_array_operations.h"
#include "../shared_fht_functions/double_array_operations.h"


#define VALID_INPUTS 0
#define INVALID_INPUTS 1
#define EPS_TOLERANCE 0.0
#define MAX_THREADS 8


/*!
 * # rbfFeatureGenDouble_
 *
 * Generates features for the input array and stores them in outputArray.
 *
 * ## Args:
 *
 * + `cArray` Pointer to the first element of the input array.
 * Must be a 3d array (N x D x C). C must be a power of 2.
 * + `radem` Pointer to first element of int8_t stack of diagonal.
 * arrays. Must be shape (3 x D x C).
 * + `chiArr` Pointer to first element of diagonal array to ensure
 * correct marginals. Must be of shape numFreqs.
 * + `outputArray` Pointer to first element of output array. Must
 * be a 2d array (N x 2 * numFreqs).
 * + `rbfNormConstant` A value by which all outputs are multipled.
 * Should be beta hparam * sqrt(1 / numFreqs). Is calculated by
 * caller.
 * + `dim0` shape[0] of input array
 * + `dim1` shape[1] of input array
 * + `dim2` shape[2] of input array
 * + `numFreqs` (numRFFs / 2) -- the number of frequencies to sample.
 * + `numThreads` The number of threads to use.
 */
const char *rbfFeatureGenDouble_(double *cArray, int8_t *radem,
                double *chiArr, double *outputArray,
                double rbfNormConstant,
                int dim0, int dim1, int dim2,
                int numFreqs, int numThreads){
    if (numThreads > dim0)
        numThreads = dim0;

    struct ThreadRBFDoubleArgs *th_args = malloc(numThreads * sizeof(struct ThreadRBFDoubleArgs));
    if (th_args == NULL){
        PyErr_SetString(PyExc_ValueError, "Memory allocation unsuccessful! Your system may be out of memory and "
            "is likely about to crash.");
        return "error";
    }
    //Note the variable length arrays, which are fine with gcc BUT may be a problem for some older
    //C++ compilers.
    int i, threadFlags[numThreads];
    int iret[numThreads];
    void *retval[numThreads];
    pthread_t thread_id[numThreads];
    
    int chunkSize = (dim0 + numThreads - 1) / numThreads;

    for (i=0; i < numThreads; i++){
        th_args[i].startPosition = i * chunkSize;
        th_args[i].endPosition = (i + 1) * chunkSize;
        if (th_args[i].endPosition > dim0)
            th_args[i].endPosition = dim0;
        th_args[i].arrayStart = cArray;
        th_args[i].dim1 = dim1;
        th_args[i].dim2 = dim2;
        th_args[i].rademArray = radem;
        th_args[i].chiArr = chiArr;
        th_args[i].outputArray = outputArray;
        th_args[i].numFreqs = numFreqs;
        th_args[i].rbfNormConstant = rbfNormConstant;
    }
    
    for (i=0; i < numThreads; i++){
        iret[i] = pthread_create(&thread_id[i], NULL, ThreadRBFGenDouble, &th_args[i]);
        if (iret[i]){
            PyErr_SetString(PyExc_ValueError, "fastHadamardTransform failed to create a thread!");
            free(th_args);
            return "error";
        }
    }
    for (i=0; i < numThreads; i++)
        threadFlags[i] = pthread_join(thread_id[i], &retval[i]);
    
    for (i=0; i < numThreads; i++){
        if (threadFlags[i] != 0){
            free(th_args);
            return "error";
        }
    }
    free(th_args);
    return "no_error";
}




/*!
 * # rbfDoubleGrad_
 *
 * Generates features for the input array and stores them in outputArray.
 *
 * ## Args:
 *
 * + `cArray` Pointer to the first element of the input array.
 * Must be a 3d array (N x D x C). C must be a power of 2.
 * + `radem` Pointer to first element of int8_t stack of diagonal.
 * arrays. Must be shape (3 x D x C).
 * + `chiArr` Pointer to first element of diagonal array to ensure
 * correct marginals. Must be of shape numFreqs.
 * + `outputArray` Pointer to first element of output array. Must
 * be a 2d array (N x 2 * numFreqs).
 * + `gradientArray` Pointer to the first element of gradient array.
 * Must be a 2d array (N x 2 * numFreqs).
 * + `rbfNormConstant` A value by which all outputs are multipled.
 * Should be beta hparam * sqrt(1 / numFreqs). Is calculated by
 * caller.
 * + `sigma` The lengthscale hyperparameter.
 * + `dim0` shape[0] of input array
 * + `dim1` shape[1] of input array
 * + `dim2` shape[2] of input array
 * + `numFreqs` (numRFFs / 2) -- the number of frequencies to sample.
 * + `numThreads` The number of threads to use.
 */
const char *rbfDoubleGrad_(double *cArray, int8_t *radem,
                double *chiArr, double *outputArray,
                double *gradientArray,
                double rbfNormConstant, double sigma,
                int dim0, int dim1, int dim2,
                int numFreqs, int numThreads){
    if (numThreads > dim0)
        numThreads = dim0;

    struct ThreadRBFDoubleGradArgs *th_args = malloc(numThreads * sizeof(struct ThreadRBFDoubleGradArgs));
    if (th_args == NULL){
        PyErr_SetString(PyExc_ValueError, "Memory allocation unsuccessful!");
        return "error";
    }
    int i, threadFlags[numThreads];
    int iret[numThreads];
    void *retval[numThreads];
    pthread_t thread_id[numThreads];
    
    int chunkSize = (dim0 + numThreads - 1) / numThreads;

    for (i=0; i < numThreads; i++){
        th_args[i].startPosition = i * chunkSize;
        th_args[i].endPosition = (i + 1) * chunkSize;
        if (th_args[i].endPosition > dim0)
            th_args[i].endPosition = dim0;
        th_args[i].arrayStart = cArray;
        th_args[i].dim1 = dim1;
        th_args[i].dim2 = dim2;
        th_args[i].rademArray = radem;
        th_args[i].chiArr = chiArr;
        th_args[i].outputArray = outputArray;
        th_args[i].numFreqs = numFreqs;
        th_args[i].rbfNormConstant = rbfNormConstant;

        th_args[i].gradientArray = gradientArray;
        th_args[i].sigma = sigma;
    }
    
    for (i=0; i < numThreads; i++){
        iret[i] = pthread_create(&thread_id[i], NULL, ThreadRBFDoubleGrad, &th_args[i]);
        if (iret[i]){
            PyErr_SetString(PyExc_ValueError, "fastHadamardTransform failed to create a thread!");
            free(th_args);
            return "error";
        }
    }
    for (i=0; i < numThreads; i++)
        threadFlags[i] = pthread_join(thread_id[i], &retval[i]);
    
    for (i=0; i < numThreads; i++){
        if (threadFlags[i] != 0){
            free(th_args);
            return "error";
        }
    }
    free(th_args);
    return "no_error";
}



/*!
 * # ardDoubleGrad_
 *
 * Performs gradient-only calculations for the mini ARD kernel.
 *
 * ## Args:
 *
 * + `inputX` Pointer to the first element of the raw input data,
 * an (N x D) array.
 * + `randomFeatures` Pointer to first element of the array in which
 * random features will be stored, an (N x 2 * C) array.
 * + `precompWeights` Pointer to first element of the array containing
 * the precomputed weights, a (C x D) array.
 * + `sigmaMap` Pointer to first element of the array containing a mapping
 * from positions to lengthscales, a (D) array.
 * + `sigmaVals` Pointer to first element of shape (D) array containing the
 * per-feature lengthscales.
 * + `gradient` Pointer to first element of the array in which the gradient
 * will be stored, an (N x 2 * C) array.
 * + `dim0` shape[0] of input X
 * + `dim1` shape[1] of input X
 * + `numLengthscales` shape[2] of gradient
 * + `numFreqs` shape[0] of precompWeights
 * + `rbfNormConstant` A value by which all outputs are multipled.
 * Should be beta hparam * sqrt(1 / numFreqs). Is calculated by
 * caller.
 * + `numThreads` The number of threads to use.
 */
const char *ardDoubleGrad_(double *inputX, double *randomFeatures,
        double *precompWeights, int32_t *sigmaMap, double *sigmaVals,
        double *gradient, int dim0, int dim1, int numLengthscales,
        int numFreqs, double rbfNormConstant, int numThreads){
    if (numThreads > dim0)
        numThreads = dim0;

    struct ThreadARDDoubleGradArgs *th_args = malloc(numThreads * sizeof(struct ThreadARDDoubleGradArgs));
    if (th_args == NULL){
        PyErr_SetString(PyExc_ValueError, "Memory allocation unsuccessful!");
        return "error";
    }
    int i, threadFlags[numThreads];
    int iret[numThreads];
    void *retval[numThreads];
    pthread_t thread_id[numThreads];
    
    int chunkSize = (dim0 + numThreads - 1) / numThreads;

    for (i=0; i < numThreads; i++){
        th_args[i].startPosition = i * chunkSize;
        th_args[i].endPosition = (i + 1) * chunkSize;
        if (th_args[i].endPosition > dim0)
            th_args[i].endPosition = dim0;
        th_args[i].inputX = inputX;
        th_args[i].dim1 = dim1;
        th_args[i].precompWeights = precompWeights;
        th_args[i].randomFeats = randomFeatures;
        th_args[i].gradientArray = gradient;
        th_args[i].sigmaMap = sigmaMap;
        th_args[i].sigmaVals = sigmaVals;
        th_args[i].numFreqs = numFreqs;
        th_args[i].numLengthscales = numLengthscales;
        th_args[i].rbfNormConstant = rbfNormConstant;
    }
    
    for (i=0; i < numThreads; i++){
        iret[i] = pthread_create(&thread_id[i], NULL, ThreadARDDoubleGrad, &th_args[i]);
        if (iret[i]){
            PyErr_SetString(PyExc_ValueError, "fastHadamardTransform failed to create a thread!");
            free(th_args);
            return "error";
        }
    }
    for (i=0; i < numThreads; i++)
        threadFlags[i] = pthread_join(thread_id[i], &retval[i]);
    
    for (i=0; i < numThreads; i++){
        if (threadFlags[i] != 0){
            free(th_args);
            return "error";
        }
    }
    free(th_args);
    return "no_error";
}



/*!
 * # ThreadRBFGenDouble
 *
 * Performs the RBF feature gen operation for one thread for a chunk of
 * the input array from startRow through endRow (each thread
 * works on its own group of rows).
 *
 * ## Args:
 *
 * + `rowArgs` a void pointer to a ThreadRBFArgs struct.
 * Contains all the arrays and info (e.g. startRow, endRow) needed
 * to process the chunk of the array belonging to this thread.
 */
void *ThreadRBFGenDouble(void *rowArgs){
    struct ThreadRBFDoubleArgs *thArgs = (struct ThreadRBFDoubleArgs *)rowArgs;
    int rowSize = thArgs->dim1 * thArgs->dim2;

    doubleMultiplyByDiagonalRademacherMat(thArgs->arrayStart,
                    thArgs->rademArray,
                    thArgs->dim1, thArgs->dim2, 
                    thArgs->startPosition, thArgs->endPosition);
    doubleTransformRows3D(thArgs->arrayStart, thArgs->startPosition, 
                    thArgs->endPosition, thArgs->dim1, thArgs->dim2);

    doubleMultiplyByDiagonalRademacherMat(thArgs->arrayStart,
                    thArgs->rademArray + rowSize,
                    thArgs->dim1, thArgs->dim2, 
                    thArgs->startPosition, thArgs->endPosition);
    doubleTransformRows3D(thArgs->arrayStart, thArgs->startPosition, 
                    thArgs->endPosition, thArgs->dim1, thArgs->dim2);
    
    doubleMultiplyByDiagonalRademacherMat(thArgs->arrayStart,
                    thArgs->rademArray + 2 * rowSize,
                    thArgs->dim1, thArgs->dim2, 
                    thArgs->startPosition, thArgs->endPosition);
    doubleTransformRows3D(thArgs->arrayStart, thArgs->startPosition, 
                    thArgs->endPosition, thArgs->dim1, thArgs->dim2);
    rbfDoubleFeatureGenLastStep_(thArgs->arrayStart, thArgs->chiArr,
                    thArgs->outputArray, thArgs->rbfNormConstant,
                    thArgs->startPosition, thArgs->endPosition,
                    thArgs->dim1, thArgs->dim2, thArgs->numFreqs);
    return NULL;
}


/*!
 * # ThreadRBFDoubleGrad
 *
 * Performs the RBF feature gen AND gradient operation for one thread
 * for a chunk of the input array from startRow through endRow (each thread
 * works on its own group of rows).
 *
 * ## Args:
 *
 * + `rowArgs` a void pointer to a ThreadRBFGradArgs struct.
 * Contains all the arrays and info (e.g. startRow, endRow) needed
 * to process the chunk of the array belonging to this thread.
 */
void *ThreadRBFDoubleGrad(void *rowArgs){
    struct ThreadRBFDoubleGradArgs *thArgs = (struct ThreadRBFDoubleGradArgs *)rowArgs;
    int rowSize = thArgs->dim1 * thArgs->dim2;

    doubleMultiplyByDiagonalRademacherMat(thArgs->arrayStart,
                    thArgs->rademArray,
                    thArgs->dim1, thArgs->dim2, 
                    thArgs->startPosition, thArgs->endPosition);
    doubleTransformRows3D(thArgs->arrayStart, thArgs->startPosition, 
                    thArgs->endPosition, thArgs->dim1, thArgs->dim2);

    doubleMultiplyByDiagonalRademacherMat(thArgs->arrayStart,
                    thArgs->rademArray + rowSize,
                    thArgs->dim1, thArgs->dim2, 
                    thArgs->startPosition, thArgs->endPosition);
    doubleTransformRows3D(thArgs->arrayStart, thArgs->startPosition, 
                    thArgs->endPosition, thArgs->dim1, thArgs->dim2);
    
    doubleMultiplyByDiagonalRademacherMat(thArgs->arrayStart,
                    thArgs->rademArray + 2 * rowSize,
                    thArgs->dim1, thArgs->dim2, 
                    thArgs->startPosition, thArgs->endPosition);
    doubleTransformRows3D(thArgs->arrayStart, thArgs->startPosition, 
                    thArgs->endPosition, thArgs->dim1, thArgs->dim2);
    rbfDoubleGradLastStep_(thArgs->arrayStart, thArgs->chiArr,
                    thArgs->outputArray, thArgs->gradientArray,
                    thArgs->rbfNormConstant, thArgs->sigma,
                    thArgs->startPosition, thArgs->endPosition,
                    thArgs->dim1, thArgs->dim2, thArgs->numFreqs);
    return NULL;
}


/*!
 * # ThreadARDDoubleGrad
 *
 * Performs ARD gradient-only calculations using pregenerated
 * features and weights.
 *
 * ## Args:
 *
 * + `rowArgs` a void pointer to a ThreadARDGradArgs struct.
 * Contains all the arrays and info (e.g. startRow, endRow) needed
 * to process the chunk of the array belonging to this thread.
 */
void *ThreadARDDoubleGrad(void *rowArgs){
    struct ThreadARDDoubleGradArgs *thArgs = (struct ThreadARDDoubleGradArgs *)rowArgs;
    ardDoubleGradCalcs_(thArgs->inputX, thArgs->randomFeats,
                    thArgs->precompWeights, thArgs->sigmaMap,
                    thArgs->sigmaVals,
                    thArgs->gradientArray, thArgs->startPosition,
                    thArgs->endPosition, thArgs->dim1,
                    thArgs->numLengthscales,
                    thArgs->rbfNormConstant,
                    thArgs->numFreqs);
    return NULL;
}



/*!
 * # rbfDoubleFeatureGenLastStep_
 *
 * Performs the last steps in RBF feature generation.
 *
 * ## Args:
 *
 * + `xArray` Pointer to the first element of the input array.
 * + `chiArray` Pointer to first element of diagonal array to ensure
 * correct marginals.
 * + `outputArray` Pointer to first element of output array.
 * + `rbfNormConstant` A value by which all outputs are multipled.
 * + `startRow` The starting row for this thread to work on.
 * + `endRow` The ending row for this thread to work on.
 * + `dim1` shape[1] of input array
 * + `dim2` shape[2] of input array
 * + `numFreqs` (numRFFs / 2) -- the number of frequencies to sample.
 */
void rbfDoubleFeatureGenLastStep_(double *xArray, double *chiArray,
        double *outputArray, double normConstant,
        int startRow, int endRow, int dim1,
        int dim2, int numFreqs){
    int i, j;
    int elementsPerRow = dim1 * dim2;
    double *xElement;
    double *outputElement;
    double outputVal;

    for (i=startRow; i < endRow; i++){
        xElement = xArray + i * elementsPerRow;
        outputElement = outputArray + i * 2 * numFreqs;
        for (j=0; j < numFreqs; j++){
            outputVal = *xElement * chiArray[j];
            *outputElement = normConstant * cos(outputVal);
            outputElement[numFreqs] = normConstant * sin(outputVal);
            outputElement++;
            xElement++;
        }
    }
}



/*!
 * # rbfDoubleGradLastStep_
 *
 * Performs the last steps in RBF feature + gradient calcs.
 *
 * ## Args:
 *
 * + `xArray` Pointer to the first element of the input array.
 * + `chiArray` Pointer to first element of diagonal array to ensure
 * correct marginals.
 * + `outputArray` Pointer to first element of output array.
 * + `rbfNormConstant` A value by which all outputs are multipled.
 * + `startRow` The starting row for this thread to work on.
 * + `endRow` The ending row for this thread to work on.
 * + `dim1` shape[1] of input array
 * + `dim2` shape[2] of input array
 * + `numFreqs` (numRFFs / 2) -- the number of frequencies to sample.
 */
void rbfDoubleGradLastStep_(double *xArray, double *chiArray,
        double *outputArray, double *gradientArray,
        double normConstant, double sigma,
        int startRow, int endRow, int dim1,
        int dim2, int numFreqs){
    int i, j;
    int elementsPerRow = dim1 * dim2;
    double *xElement;
    double *outputElement, *gradientElement;
    double outputVal, cosVal, sinVal;

    for (i=startRow; i < endRow; i++){
        xElement = xArray + i * elementsPerRow;
        outputElement = outputArray + i * 2 * numFreqs;
        gradientElement = gradientArray + i * 2 * numFreqs;
        for (j=0; j < numFreqs; j++){
            outputVal = *xElement * chiArray[j];
            cosVal = cos(outputVal * sigma) * normConstant;
            sinVal = sin(outputVal * sigma) * normConstant;

            *outputElement = cosVal;
            outputElement[numFreqs] = sinVal;
            *gradientElement = -sinVal * outputVal;
            gradientElement[numFreqs] = cosVal * outputVal;

            outputElement++;
            gradientElement++;
            xElement++;
        }
    }
}




/*!
 * # ardDoubleGradCalcs_
 *
 * Performs the key calculations for the miniARD gradient.
 *
 * ## Args:
 *
 * + `inputX` Pointer to the first element of the input array.
 * + `randomFeatures` Pointer to first element of random feature array.
 * + `precompWeights` Pointer to first element of precomputed weights.
 * + `sigmaMap` Pointer to first element of the array containing a
 * mapping from positions to lengthscales.
 * + `sigmaVals` Pointer to first element of shape (D) array containing the
 * per-feature lengthscales.
 * + `gradient` Pointer to the output array.
 * + `startRow` The starting row for this thread to work on.
 * + `endRow` The ending row for this thread to work on.
 * + `dim1` shape[1] of input array
 * + `numLengthscales` shape[2] of gradient
 * + `rbfNormConstant` A value by which all outputs are multipled.
 * Should be beta hparam * sqrt(1 / numFreqs). Is calculated by
 * caller.
 * + `numFreqs` (numRFFs / 2) -- the number of frequencies to sample.
 */
void ardDoubleGradCalcs_(double *inputX, double *randomFeatures,
        double *precompWeights, int32_t *sigmaMap, double *sigmaVals,
        double *gradient, int startRow, int endRow, int dim1,
        int numLengthscales, double rbfNormConstant,
        int numFreqs){
    int i, j, k;
    int gradIncrement = numFreqs * numLengthscales;
    double *xElement, *precompWeight, dotProd;
    double *gradientElement, *randomFeature;
    double gradVal, sinVal, cosVal, rfSum;

    xElement = inputX + startRow * dim1;
    gradientElement = gradient + startRow * 2 * gradIncrement;
    randomFeature = randomFeatures + startRow * numFreqs * 2;

    for (i=startRow; i < endRow; i++){
        precompWeight = precompWeights;

        for (j=0; j < numFreqs; j++){
            rfSum = 0;

            for (k=0; k < dim1; k++){
                dotProd = xElement[k] * *precompWeight;
                gradientElement[sigmaMap[k]] += dotProd;
                rfSum += sigmaVals[k] * dotProd;
                precompWeight++;
            }

            cosVal = rbfNormConstant * cos(rfSum);
            sinVal = rbfNormConstant * sin(rfSum);
            *randomFeature = cosVal;
            randomFeature[numFreqs] = sinVal;

            for (k=0; k < numLengthscales; k++){
                gradVal = gradientElement[k];
                gradientElement[k] = -gradVal * sinVal;
                gradientElement[k + gradIncrement] = gradVal * cosVal;
            }
            gradientElement += numLengthscales;
            randomFeature++;
        }
        xElement += dim1;
        gradientElement += gradIncrement;
        randomFeature += numFreqs;
    }
}