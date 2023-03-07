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
 * + ThreadRBFGenDouble
 * Performs operations for a single thread of the feature generation operation.
 *
 * + ThreadRBFDoubleGrad
 * Performs operations for a single thread of the gradient / feature operation.
 * 
 * + rbfDoubleFeatureGenLastStep_
 * Performs the final operations involved in the feature generation for doubles.
 *
 * + rbfDoubleGradLastStep_
 * Performs the final operations involved in feature / gradient calc for doubles.
 */
#include <Python.h>
#include <numpy/arrayobject.h>
#include <pthread.h>
#include <math.h>
#include "double_rbf_ops.h"
#include "../float_array_operations.h"
#include "../double_array_operations.h"


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
