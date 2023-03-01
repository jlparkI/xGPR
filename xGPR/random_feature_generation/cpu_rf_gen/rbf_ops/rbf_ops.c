/*!
 * # rbf_ops.c
 *
 * This module performs all major steps involved in feature generation for
 * RBF-type kernels, which includes RBF, Matern, ARD and MiniARD (and by extension
 * the static layer kernels). Functions from array_operations are used to perform
 * the fast Hadamard transform and rademacher matrix multiplication pieces.
 * The "specialized" piece, multiplication by a diagonal matrix while performing
 * sine-cosine operations, is performed here.
 *
 * + rbfFeatureGenFloat_
 * Performs the feature generation steps on an input array of floats.
 * + rbfFeatureGenDouble_
 * Performs the feature generation steps on an input array of doubles.
 *
 * + ThreadRBFGenFloat
 * Performs operations for a single thread of the feature generation operation.
 * + ThreadRBFGenDouble
 * Performs operations for a single thread of the feature generation operation.
 * 
 * + rbfFloatFeatureGenLastStep_
 * Performs the final operations involved in the feature generation for floats.
 * + rbfDoubleFeatureGenLastStep_
 * Performs the final operations involved in the feature generation for doubles.
 */
#include <Python.h>
#include <numpy/arrayobject.h>
#include <pthread.h>
#include <math.h>
#include "rbf_ops.h"
#include "../float_array_operations.h"
#include "../double_array_operations.h"


#define VALID_INPUTS 0
#define INVALID_INPUTS 1
#define EPS_TOLERANCE 0.0
#define MAX_THREADS 8

/*!
 * # rbfFeatureGenFloat_
 *
 * Generates features for the input array and stores them in outputArray.
 *
 * ## Args:
 *
 * + `cArray` Pointer to the first element of the input array.
 * Must be a 3d array (N x D x C). C must be a power of 2.
 * + `radem` Pointer to first element of int8_t stack of diagonal.
 * arrays. Must be shape (3 x 2 * numFreqs).
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
const char *rbfFeatureGenFloat_(float *cArray, int8_t *radem,
                float *chiArr, double *outputArray,
                double rbfNormConstant,
                int dim0, int dim1, int dim2,
                int numFreqs, int numThreads){
    if (numThreads > dim0)
        numThreads = dim0;

    struct ThreadRBFFloatArgs *th_args = malloc(numThreads * sizeof(struct ThreadRBFFloatArgs));
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
        iret[i] = pthread_create(&thread_id[i], NULL, ThreadRBFGenFloat, &th_args[i]);
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
 * # rbfFeatureGenFloat_
 *
 * Generates features for the input array and stores them in outputArray.
 *
 * ## Args:
 *
 * + `cArray` Pointer to the first element of the input array.
 * Must be a 3d array (N x D x C). C must be a power of 2.
 * + `radem` Pointer to first element of int8_t stack of diagonal.
 * arrays. Must be shape (3 x 2 * numFreqs).
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
 * # ThreadRBFGenFloat
 *
 * Performs the RBF feature gen operation for one thread for a chunk of
 * the input array from startRow through endRow (each thread
 * works on its own group of rows), for arrays of floats only.
 *
 * ## Args:
 *
 * + `rowArgs` a void pointer to a ThreadRBFArgs struct.
 * Contains all the arrays and info (e.g. startRow, endRow) needed
 * to process the chunk of the array belonging to this thread.
 */
void *ThreadRBFGenFloat(void *rowArgs){
    struct ThreadRBFFloatArgs *thArgs = (struct ThreadRBFFloatArgs *)rowArgs;
    int rowSize = thArgs->dim1 * thArgs->dim2;

    floatMultiplyByDiagonalRademacherMat(thArgs->arrayStart,
                    thArgs->rademArray,
                    thArgs->dim1, thArgs->dim2, 
                    thArgs->startPosition, thArgs->endPosition);
    floatTransformRows3D(thArgs->arrayStart, thArgs->startPosition, 
                    thArgs->endPosition, thArgs->dim1, thArgs->dim2);

    floatMultiplyByDiagonalRademacherMat(thArgs->arrayStart,
                    thArgs->rademArray + rowSize,
                    thArgs->dim1, thArgs->dim2, 
                    thArgs->startPosition, thArgs->endPosition);
    floatTransformRows3D(thArgs->arrayStart, thArgs->startPosition, 
                    thArgs->endPosition, thArgs->dim1, thArgs->dim2);
    
    floatMultiplyByDiagonalRademacherMat(thArgs->arrayStart,
                    thArgs->rademArray + 2 * rowSize,
                    thArgs->dim1, thArgs->dim2, 
                    thArgs->startPosition, thArgs->endPosition);
    floatTransformRows3D(thArgs->arrayStart, thArgs->startPosition, 
                    thArgs->endPosition, thArgs->dim1, thArgs->dim2);
    rbfFloatFeatureGenLastStep_(thArgs->arrayStart, thArgs->chiArr,
                    thArgs->outputArray, thArgs->rbfNormConstant,
                    thArgs->startPosition, thArgs->endPosition,
                    thArgs->dim1, thArgs->dim2, thArgs->numFreqs);
    return NULL;
}



/*!
 * # ThreadRBFGenFloat
 *
 * Performs the RBF feature gen operation for one thread for a chunk of
 * the input array from startRow through endRow (each thread
 * works on its own group of rows), for arrays of floats only.
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
 * # rbfFloatFeatureGenLastStep_
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
void rbfFloatFeatureGenLastStep_(float *xArray, float *chiArray,
        double *outputArray, double normConstant,
        int startRow, int endRow, int dim1,
        int dim2, int numFreqs){
    int i, j;
    int elementsPerRow = dim1 * dim2;
    float *xElement, *chiElement;
    double *outputElement;
    float outputVal;

    for (i=startRow; i < endRow; i++){
        xElement = xArray + i * elementsPerRow;
        chiElement = chiArray;
        outputElement = outputArray + i * 2 * numFreqs;
        for (j=0; j < numFreqs; j++){
            outputVal = *xElement * *chiElement;
            *outputElement = normConstant * cosf(outputVal);
            outputElement[numFreqs] = normConstant * sinf(outputVal);
            outputElement++;
            chiElement++;
            xElement++;
        }
    }
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
    double *xElement, *chiElement;
    double *outputElement;
    double outputVal;

    for (i=startRow; i < endRow; i++){
        xElement = xArray + i * elementsPerRow;
        chiElement = chiArray;
        outputElement = outputArray + i * 2 * numFreqs;
        for (j=0; j < numFreqs; j++){
            outputVal = *xElement * *chiElement;
            *outputElement = normConstant * cos(outputVal);
            outputElement[numFreqs] = normConstant * sin(outputVal);
            outputElement++;
            chiElement++;
            xElement++;
        }
    }
}
