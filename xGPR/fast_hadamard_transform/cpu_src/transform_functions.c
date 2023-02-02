/*!
 * # transform_functions.c
 *
 * This module performs the fast Hadamard transform on a 3d array
 * or the structured orthogonal features (SORF) operation on a 3d
 * array. It includes the following functions:
 * + fastHadamard3dFloatArray_
 * Performs the unnormalized fast Hadamard on a 3d array of floats using multithreading.
 * + fastHadamard3dDoubleArray_
 * Performs the unnormalized fast Hadamard on a 3d array of doubles using multithreading.
 * + fastHadamard2dFloatArray_
 * Performs the unnormalized fast Hadamard on a 2d array of floats using multithreading.
 * + fastHadamard2dDoubleArray_
 * Performs the unnormalized fast Hadamard on a 2d array of doubles using multithreading.
 * 
 * + SORFFloatBlockTransform_
 * Performs the structured orthogonal features operation on an input
 * 3d array of floats using multithreading.
 * + SORFDoubleBlockTransform_
 * Performs the structured orthogonal features operation on an input
 * 3d array of doubles using multithreading.
 * 
 * + SRHTFloatBlockTransform_
 * Performs the key operations in the SRHT on an input 2d array of
 * floats using multithreading.
 * + SRHTDoubleBlockTransform_
 * Performs the key operations in the SRHT on an input 2d array of
 * doubles using multithreading.
 *
 * + ThreadSORFFloatRows3D
 * Performs operations for a single thread of the SORFFloat operation.
 * + ThreadSORFDoubleRows3D
 * Performs operations for a single thread of the SORFDouble operation.
 *
 * + ThreadSRHTFloatRows
 * Performs operations for a single thread of the SRHTFloat operation.
 * + ThreadSRHTDoubleRows
 * Performs operations for a single thread of the SRHTDouble operation.
 *
 * + ThreadTransformRows3DFloat
 * Performs operations for a single thread of the fast Hadamard for floats.
 * + ThreadTransformRows3DDouble
 * Performs operations for a single thread of the fast Hadamard for doubles.
 * 
 * + ThreadTransformRows2DFloat
 * Performs operations for a single thread of the fast Hadamard 2d for floats.
 * + ThreadTransformRows2DDouble
 * Performs operations for a single thread of the fast Hadamard 2d for doubles.
 */
#include <Python.h>
#include <numpy/arrayobject.h>
#include <pthread.h>
#include "thread_args.h"
#include "transform_functions.h"
#include "float_array_operations.h"
#include "double_array_operations.h"


#define VALID_INPUTS 0
#define INVALID_INPUTS 1
#define EPS_TOLERANCE 0.0
#define MAX_THREADS 8

/*!
 * # fastHadamard3dFloatArray_
 *
 * Performs an unnormalized Hadamard transform along the last
 * dimension of an input 3d array, floats only. The transform is performed
 * in place so nothing is returned.
 *
 * ## Args:
 *
 * + `Z` Pointer to the first element of the array to be
 * modified. Must be a 3d array (N x D x C). C must be a power of 2.
 * + `zDim0` The first dimension of Z (N).
 * + `zDim1` The second dimension of Z (D)
 * + `zDim2` The third dimension of Z (C)
 * + `numThreads` The number of threads to use.
 */
const char *fastHadamard3dFloatArray_(float *Z, int zDim0, int zDim1, int zDim2,
                        int numThreads)
{
    struct Thread3DFloatArrayArgs *th_args = malloc(numThreads * sizeof(struct Thread3DFloatArrayArgs));
    if (th_args == NULL){
        PyErr_SetString(PyExc_ValueError, "Memory allocation unsuccessful! Your system may be out of memory and "
            "is likely about to crash.");
        return "error";
    }
    //Note the variable length arrays, which are fine with gcc BUT may be a problem for some older
    //C++ compilers.
    int i, threadFlags[numThreads];
    void *retval[numThreads];
    int iret[numThreads];
    pthread_t thread_id[numThreads];

    int chunkSize = (zDim0 + numThreads - 1) / numThreads;

    for (i=0; i < numThreads; i++){
        th_args[i].startPosition = i * chunkSize;
        th_args[i].endPosition = (i + 1) * chunkSize;
        if (th_args[i].endPosition > zDim0)
            th_args[i].endPosition = zDim0;
        th_args[i].arrayStart = Z;
        th_args[i].dim1 = zDim1;
        th_args[i].dim2 = zDim2;
    }

    for (i=0; i < numThreads; i++){
        iret[i] = pthread_create(&thread_id[i], NULL, ThreadTransformRows3DFloat, &th_args[i]);
        if (iret[i]){
            PyErr_SetString(PyExc_ValueError, "fastHadamardTransform failed to create a thread!");
            return "error";
        }
    }
    for (i=0; i < numThreads; i++)
        threadFlags[i] = pthread_join(thread_id[i], &retval[i]);
    
    free(th_args);
    return "no_error";
}



/*!
 * # fastHadamard3dDoubleArray_
 *
 * Performs an unnormalized Hadamard transform along the last
 * dimension of an input 3d array, doubles only. The transform is performed
 * in place so nothing is returned.
 *
 * ## Args:
 *
 * + `Z` Pointer to the first element of the array to be
 * modified. Must be a 3d array (N x D x C). C must be a power of 2.
 * + `zDim0` The first dimension of Z (N).
 * + `zDim1` The second dimension of Z (D)
 * + `zDim2` The third dimension of Z (C)
 * + `numThreads` The number of threads to use.
 */
const char *fastHadamard3dDoubleArray_(double *Z, int zDim0, int zDim1, int zDim2,
                        int numThreads)
{
    struct Thread3DDoubleArrayArgs *th_args = malloc(numThreads * sizeof(struct Thread3DDoubleArrayArgs));
    if (th_args == NULL){
        PyErr_SetString(PyExc_ValueError, "Memory allocation unsuccessful! Your system may be out of memory and "
            "is likely about to crash.");
        return "error";
    }
    //Note the variable length arrays, which are fine with gcc BUT may be a problem for some older
    //C++ compilers.
    int i, threadFlags[numThreads];
    void *retval[numThreads];
    int iret[numThreads];
    pthread_t thread_id[numThreads];

    int chunkSize = (zDim0 + numThreads - 1) / numThreads;

    for (i=0; i < numThreads; i++){
        th_args[i].startPosition = i * chunkSize;
        th_args[i].endPosition = (i + 1) * chunkSize;
        if (th_args[i].endPosition > zDim0)
            th_args[i].endPosition = zDim0;
        th_args[i].arrayStart = Z;
        th_args[i].dim1 = zDim1;
        th_args[i].dim2 = zDim2;
    }

    for (i=0; i < numThreads; i++){
        iret[i] = pthread_create(&thread_id[i], NULL, ThreadTransformRows3DDouble, &th_args[i]);
        if (iret[i]){
            PyErr_SetString(PyExc_ValueError, "fastHadamardTransform failed to create a thread!");
            return "error";
        }
    }
    for (i=0; i < numThreads; i++)
        threadFlags[i] = pthread_join(thread_id[i], &retval[i]);
    
    free(th_args);
    return "no_error";
}





/*!
 * # fastHadamard2dFloatArray_
 *
 * Performs an unnormalized Hadamard transform along the last
 * dimension of an input 2d array, floats only. The transform is performed
 * in place so nothing is returned.
 *
 * ## Args:
 *
 * + `Z` Pointer to the first element of the array to be
 * modified. Must be a 2d array (N x C). C must be a power of 2.
 * + `zDim0` The first dimension of Z (N).
 * + `zDim1` The second dimension of Z (C)
 * + `numThreads` The number of threads to use.
 */
const char *fastHadamard2dFloatArray_(float *Z, int zDim0, int zDim1,
                        int numThreads)
{
    struct Thread3DFloatArrayArgs *th_args = malloc(numThreads * sizeof(struct Thread3DFloatArrayArgs));
    if (th_args == NULL){
        PyErr_SetString(PyExc_ValueError, "Memory allocation unsuccessful! Your system may be out of memory and "
            "is likely about to crash.");
        return "error";
    }
    //Note the variable length arrays, which are fine with gcc BUT may be a problem for some older
    //C++ compilers.
    int i, threadFlags[numThreads];
    void *retval[numThreads];
    int iret[numThreads];
    pthread_t thread_id[numThreads];

    int chunkSize = (zDim0 + numThreads - 1) / numThreads;

    for (i=0; i < numThreads; i++){
        th_args[i].startPosition = i * chunkSize;
        th_args[i].endPosition = (i + 1) * chunkSize;
        if (th_args[i].endPosition > zDim0)
            th_args[i].endPosition = zDim0;
        th_args[i].arrayStart = Z;
        th_args[i].dim1 = zDim1;
        //th_args[i].dim2 is not used since this is a 2d array; set to 1.
        th_args[i].dim2 = 1;
    }

    for (i=0; i < numThreads; i++){
        iret[i] = pthread_create(&thread_id[i], NULL, ThreadTransformRows2DFloat, &th_args[i]);
        if (iret[i]){
            PyErr_SetString(PyExc_ValueError, "fastHadamardTransform failed to create a thread!");
            return "error";
        }
    }
    for (i=0; i < numThreads; i++)
        threadFlags[i] = pthread_join(thread_id[i], &retval[i]);
    
    free(th_args);
    return "no_error";
}




/*!
 * # fastHadamard2dDoubleArray_
 *
 * Performs an unnormalized Hadamard transform along the last
 * dimension of an input 2d array, floats only. The transform is performed
 * in place so nothing is returned.
 *
 * ## Args:
 *
 * + `Z` Pointer to the first element of the array to be
 * modified. Must be a 2d array (N x C). C must be a power of 2.
 * + `zDim0` The first dimension of Z (N).
 * + `zDim1` The second dimension of Z (C)
 * + `numThreads` The number of threads to use.
 */
const char *fastHadamard2dDoubleArray_(double *Z, int zDim0, int zDim1,
                        int numThreads)
{
    struct Thread3DDoubleArrayArgs *th_args = malloc(numThreads * sizeof(struct Thread3DDoubleArrayArgs));
    if (th_args == NULL){
        PyErr_SetString(PyExc_ValueError, "Memory allocation unsuccessful! Your system may be out of memory and "
            "is likely about to crash.");
        return "error";
    }
    //Note the variable length arrays, which are fine with gcc BUT may be a problem for some older
    //C++ compilers.
    int i, threadFlags[numThreads];
    void *retval[numThreads];
    int iret[numThreads];
    pthread_t thread_id[numThreads];

    int chunkSize = (zDim0 + numThreads - 1) / numThreads;

    for (i=0; i < numThreads; i++){
        th_args[i].startPosition = i * chunkSize;
        th_args[i].endPosition = (i + 1) * chunkSize;
        if (th_args[i].endPosition > zDim0)
            th_args[i].endPosition = zDim0;
        th_args[i].arrayStart = Z;
        th_args[i].dim1 = zDim1;
        //th_args[i].dim2 is not used since this is a 2d array; set to 1.
        th_args[i].dim2 = 1;
    }

    for (i=0; i < numThreads; i++){
        iret[i] = pthread_create(&thread_id[i], NULL, ThreadTransformRows2DDouble, &th_args[i]);
        if (iret[i]){
            PyErr_SetString(PyExc_ValueError, "fastHadamardTransform failed to create a thread!");
            return "error";
        }
    }
    for (i=0; i < numThreads; i++)
        threadFlags[i] = pthread_join(thread_id[i], &retval[i]);
    
    free(th_args);
    return "no_error";
}




/*!
 * # SORFFloatBlockTransform_
 *
 * Performs the following operation along the last dimension
 * of the input array Z: H D1 H D2 H D3, where H is a normalized
 * Hadamard transform and D1, D2, D3 are diagonal arrays.
 *
 * ## Args:
 *
 * + `Z` Pointer to the first element of the array to be
 * modified. Must be a 3d array (N x D x C). C must be a power of 2.
 * + `radem` A stack of diagonal arrays of shape (3 x D x C).
 * + `zDim0` The first dimension of Z (N).
 * + `zDim1` The second dimension of Z (D)
 * + `zDim2` The third dimension of Z (C)
 * + `numThreads` The number of threads to use.
 */
const char *SORFFloatBlockTransform_(float *Z, int8_t *radem,
            int zDim0, int zDim1, int zDim2, int numThreads)
{
    struct ThreadSORFFloatArrayArgs *th_args = malloc(numThreads * sizeof(struct ThreadSORFFloatArrayArgs));
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
    
    int chunkSize = (zDim0 + numThreads - 1) / numThreads;

    for (i=0; i < numThreads; i++){
        th_args[i].startPosition = i * chunkSize;
        th_args[i].endPosition = (i + 1) * chunkSize;
        if (th_args[i].endPosition > zDim0)
            th_args[i].endPosition = zDim0;
        th_args[i].arrayStart = Z;
        th_args[i].dim1 = zDim1;
        th_args[i].dim2 = zDim2;
        th_args[i].rademArray = radem;
    }
    
    for (i=0; i < numThreads; i++){
        iret[i] = pthread_create(&thread_id[i], NULL, ThreadSORFFloatRows3D, &th_args[i]);
        if (iret[i]){
            PyErr_SetString(PyExc_ValueError, "fastHadamardTransform failed to create a thread!");
            return "error";
        }
    }
    for (i=0; i < numThreads; i++)
        threadFlags[i] = pthread_join(thread_id[i], &retval[i]);
    
    free(th_args);
    return "no_error";
}


/*!
 * # SORFDoubleBlockTransform_
 *
 * Performs the following operation along the last dimension
 * of the input array Z: H D1 H D2 H D3, where H is a normalized
 * Hadamard transform and D1, D2, D3 are diagonal arrays.
 *
 * ## Args:
 *
 * + `Z` Pointer to the first element of the array to be
 * modified. Must be a 3d array (N x D x C). C must be a power of 2.
 * + `radem` A stack of diagonal arrays of shape (3 x D x C).
 * + `zDim0` The first dimension of Z (N).
 * + `zDim1` The second dimension of Z (D)
 * + `zDim2` The third dimension of Z (C)
 * + `numThreads` The number of threads to use.
 */
const char *SORFDoubleBlockTransform_(double *Z, int8_t *radem,
            int zDim0, int zDim1, int zDim2, int numThreads)
{
    struct ThreadSORFDoubleArrayArgs *th_args = malloc(numThreads * sizeof(struct ThreadSORFDoubleArrayArgs));
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
    
    int chunkSize = (zDim0 + numThreads - 1) / numThreads;

    for (i=0; i < numThreads; i++){
        th_args[i].startPosition = i * chunkSize;
        th_args[i].endPosition = (i + 1) * chunkSize;
        if (th_args[i].endPosition > zDim0)
            th_args[i].endPosition = zDim0;
        th_args[i].arrayStart = Z;
        th_args[i].dim1 = zDim1;
        th_args[i].dim2 = zDim2;
        th_args[i].rademArray = radem;
    }
    
    for (i=0; i < numThreads; i++){
        iret[i] = pthread_create(&thread_id[i], NULL, ThreadSORFDoubleRows3D, &th_args[i]);
        if (iret[i]){
            PyErr_SetString(PyExc_ValueError, "fastHadamardTransform failed to create a thread!");
            return "error";
        }
    }
    for (i=0; i < numThreads; i++)
        threadFlags[i] = pthread_join(thread_id[i], &retval[i]);
    
    free(th_args);
    return "no_error";
}







/*!
 * # SRHTFloatBlockTransform_
 *
 * Performs the following operation along the last dimension
 * of the input array Z: H D1, where H is a normalized
 * Hadamard transform and D1 is a diagonal array.
 *
 * ## Args:
 *
 * + `Z` Pointer to the first element of the array to be
 * modified. Must be a 2d array (N x C). C must be a power of 2.
 * + `radem` A diagonal array of shape (C).
 * + `zDim0` The first dimension of Z (N).
 * + `zDim1` The second dimension of Z (C)
 * + `numThreads` The number of threads to use.
 */
const char *SRHTFloatBlockTransform_(float *Z, int8_t *radem,
            int zDim0, int zDim1, int numThreads)
{
    struct ThreadSORFFloatArrayArgs *th_args = malloc(numThreads * sizeof(struct ThreadSORFFloatArrayArgs));
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
    
    int chunkSize = (zDim0 + numThreads - 1) / numThreads;

    for (i=0; i < numThreads; i++){
        th_args[i].startPosition = i * chunkSize;
        th_args[i].endPosition = (i + 1) * chunkSize;
        if (th_args[i].endPosition > zDim0)
            th_args[i].endPosition = zDim0;
        th_args[i].arrayStart = Z;
        th_args[i].dim1 = zDim1;
        //dim2 is not used here since this is a 2d array; set to 1
        th_args[i].dim2 = 1;
        th_args[i].rademArray = radem;
    }
    
    for (i=0; i < numThreads; i++){
        iret[i] = pthread_create(&thread_id[i], NULL, ThreadSRHTFloatRows2D, &th_args[i]);
        if (iret[i]){
            PyErr_SetString(PyExc_ValueError, "fastHadamardTransform failed to create a thread!");
            return "error";
        }
    }
    for (i=0; i < numThreads; i++)
        threadFlags[i] = pthread_join(thread_id[i], &retval[i]);
    
    free(th_args);
    return "no_error";
}



/*!
 * # SRHTDoubleBlockTransform_
 *
 * Performs the following operation along the last dimension
 * of the input array Z: H D1, where H is a normalized
 * Hadamard transform and D1 is a diagonal array.
 *
 * ## Args:
 *
 * + `Z` Pointer to the first element of the array to be
 * modified. Must be a 2d array (N x C). C must be a power of 2.
 * + `radem` A diagonal array of shape (C).
 * + `zDim0` The first dimension of Z (N).
 * + `zDim1` The second dimension of Z (C)
 * + `numThreads` The number of threads to use.
 */
const char *SRHTDoubleBlockTransform_(double *Z, int8_t *radem,
            int zDim0, int zDim1, int numThreads)
{
    struct ThreadSORFDoubleArrayArgs *th_args = malloc(numThreads * sizeof(struct ThreadSORFDoubleArrayArgs));
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
    
    int chunkSize = (zDim0 + numThreads - 1) / numThreads;

    for (i=0; i < numThreads; i++){
        th_args[i].startPosition = i * chunkSize;
        th_args[i].endPosition = (i + 1) * chunkSize;
        if (th_args[i].endPosition > zDim0)
            th_args[i].endPosition = zDim0;
        th_args[i].arrayStart = Z;
        th_args[i].dim1 = zDim1;
        //dim2 is not used here since this is a 2d array; set to 1
        th_args[i].dim2 = 1;
        th_args[i].rademArray = radem;
    }
    
    for (i=0; i < numThreads; i++){
        iret[i] = pthread_create(&thread_id[i], NULL, ThreadSRHTDoubleRows2D, &th_args[i]);
        if (iret[i]){
            PyErr_SetString(PyExc_ValueError, "fastHadamardTransform failed to create a thread!");
            return "error";
        }
    }
    for (i=0; i < numThreads; i++)
        threadFlags[i] = pthread_join(thread_id[i], &retval[i]);
    
    free(th_args);
    return "no_error";
}





/*!
 * # ThreadSORFFloatRows3D
 *
 * Performs the SORF operation for one thread for a chunk of
 * the input array Z from startRow through endRow (each thread
 * works on its own group of rows), for arrays of floats only.
 *
 * ## Args:
 *
 * + `rowArgs` a void pointer to a ThreadSORFArrayArgs struct.
 * Contains all the arrays and info (e.g. startRow, endRow) needed
 * to process the chunk of the array belonging to this thread.
 */
void *ThreadSORFFloatRows3D(void *rowArgs){
    struct ThreadSORFFloatArrayArgs *thArgs = (struct ThreadSORFFloatArrayArgs *)rowArgs;
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
    return NULL;
}

/*!
 * # ThreadSORFDoubleRows3D
 *
 * Performs the SORF operation for one thread for a chunk of
 * the input array Z from startRow through endRow (each thread
 * works on its own group of rows), for arrays of doubles only.
 *
 * ## Args:
 *
 * + `rowArgs` a void pointer to a ThreadSORFArrayArgs struct.
 * Contains all the arrays and info (e.g. startRow, endRow) needed
 * to process the chunk of the array belonging to this thread.
 */
void *ThreadSORFDoubleRows3D(void *rowArgs){
    struct ThreadSORFDoubleArrayArgs *thArgs = (struct ThreadSORFDoubleArrayArgs *)rowArgs;
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
    return NULL;
}






/*!
 * # ThreadSRHTFloatRows2D
 *
 * Performs the SRHT operation for one thread for a chunk of
 * the input array Z from startRow through endRow (each thread
 * works on its own group of rows), for arrays of floats only.
 *
 * ## Args:
 *
 * + `rowArgs` a void pointer to a ThreadSORFArrayArgs struct.
 * Contains all the arrays and info (e.g. startRow, endRow) needed
 * to process the chunk of the array belonging to this thread.
 */
void *ThreadSRHTFloatRows2D(void *rowArgs){
    struct ThreadSORFFloatArrayArgs *thArgs = (struct ThreadSORFFloatArrayArgs *)rowArgs;

    floatMultiplyByDiagonalRademacherMat2D(thArgs->arrayStart,
                    thArgs->rademArray, thArgs->dim1,
                    thArgs->startPosition, thArgs->endPosition);
    floatTransformRows2D(thArgs->arrayStart, thArgs->startPosition, 
                    thArgs->endPosition, thArgs->dim1);
    return NULL;
}



/*!
 * # ThreadSRHTDoubleRows2D
 *
 * Performs the SORF operation for one thread for a chunk of
 * the input array Z from startRow through endRow (each thread
 * works on its own group of rows), for arrays of doubles only.
 *
 * ## Args:
 *
 * + `rowArgs` a void pointer to a ThreadSORFArrayArgs struct.
 * Contains all the arrays and info (e.g. startRow, endRow) needed
 * to process the chunk of the array belonging to this thread.
 */
void *ThreadSRHTDoubleRows2D(void *rowArgs){
    struct ThreadSORFDoubleArrayArgs *thArgs = (struct ThreadSORFDoubleArrayArgs *)rowArgs;

    doubleMultiplyByDiagonalRademacherMat2D(thArgs->arrayStart,
                    thArgs->rademArray, thArgs->dim1,
                    thArgs->startPosition, thArgs->endPosition);
    doubleTransformRows2D(thArgs->arrayStart, thArgs->startPosition, 
                    thArgs->endPosition, thArgs->dim1);
    return NULL;
}





/*!
 * # ThreadTransformRows3DFloat
 *
 * Performs the fast Hadamard transform for one thread for a chunk of
 * the input array Z from startRow through endRow (each thread
 * works on its own group of rows), when the input is an array of floats.
 *
 * ## Args:
 *
 * + `rowArgs` a void pointer to a Thread3DFloatArrayArgs struct.
 * Contains all the arrays and info (e.g. startRow, endRow) needed
 * to process the chunk of the array belonging to this thread.
 */
void *ThreadTransformRows3DFloat(void *rowArgs){
    struct Thread3DFloatArrayArgs *thArgs = (struct Thread3DFloatArrayArgs *)rowArgs;
    floatTransformRows3D(thArgs->arrayStart, thArgs->startPosition, 
                    thArgs->endPosition, thArgs->dim1, thArgs->dim2);
    return NULL;
}




/*!
 * # ThreadTransformRows3DDouble
 *
 * Performs the fast Hadamard transform for one thread for a chunk of
 * the input array Z from startRow through endRow (each thread
 * works on its own group of rows), when the input is an array of floats.
 *
 * ## Args:
 *
 * + `rowArgs` a void pointer to a Thread3DDoubleArrayArgs struct.
 * Contains all the arrays and info (e.g. startRow, endRow) needed
 * to process the chunk of the array belonging to this thread.
 */
void *ThreadTransformRows3DDouble(void *rowArgs){
    struct Thread3DDoubleArrayArgs *thArgs = (struct Thread3DDoubleArrayArgs *)rowArgs;
    doubleTransformRows3D(thArgs->arrayStart, thArgs->startPosition, 
                    thArgs->endPosition, thArgs->dim1, thArgs->dim2);
    return NULL;
}





/*!
 * # ThreadTransformRows2DFloat
 *
 * Performs the fast Hadamard transform for one thread for a chunk of
 * the input array Z from startRow through endRow (each thread
 * works on its own group of rows), when the input is an array of floats.
 *
 * ## Args:
 *
 * + `rowArgs` a void pointer to a Thread3DFloatArrayArgs struct.
 * Contains all the arrays and info (e.g. startRow, endRow) needed
 * to process the chunk of the array belonging to this thread.
 */
void *ThreadTransformRows2DFloat(void *rowArgs){
    struct Thread3DFloatArrayArgs *thArgs = (struct Thread3DFloatArrayArgs *)rowArgs;
    floatTransformRows2D(thArgs->arrayStart, thArgs->startPosition, 
                    thArgs->endPosition, thArgs->dim1);
    return NULL;
}




/*!
 * # ThreadTransformRows2DDouble
 *
 * Performs the fast Hadamard transform for one thread for a chunk of
 * the input array Z from startRow through endRow (each thread
 * works on its own group of rows), when the input is an array of floats.
 *
 * ## Args:
 *
 * + `rowArgs` a void pointer to a Thread3DDoubleArrayArgs struct.
 * Contains all the arrays and info (e.g. startRow, endRow) needed
 * to process the chunk of the array belonging to this thread.
 */
void *ThreadTransformRows2DDouble(void *rowArgs){
    struct Thread3DDoubleArrayArgs *thArgs = (struct Thread3DDoubleArrayArgs *)rowArgs;
    doubleTransformRows2D(thArgs->arrayStart, thArgs->startPosition, 
                    thArgs->endPosition, thArgs->dim1);
    return NULL;
}
