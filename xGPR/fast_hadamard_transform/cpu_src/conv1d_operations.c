/*!
 * # conv1d_operations.c
 *
 * This module performs operations unique to the convolution
 * kernels in xGPR, essentially orthogonal random features based
 * convolution. It includes the following functions:
 *
 * + doubleConv1dPrep_
 * Performs the core fast hadamard transform based operations needed
 * for convolution with SORF. The activation functions and other steps are
 * applied by the Cython wrapper.
 *
 * + floatConv1dPrep_
 * Performs the core fast hadamard transform based operations needed
 * for convolution with SORF, but for single-precision arrays. The activation
 * functions and other steps are applied by the Cython wrapper.
 *
 * + FloatThreadConv1d
 * Called once by doubleConv1dPrep_ for each thread.
 *
 * + DoubleThreadConv1d
 * Called once by doubleConv1dPrep_ for each thread.
 *
 * Functions from float_ and double_ array_operations.c are called to
 * perform the Hadamard transform and diagonal matrix multiplications.
 */
#include <Python.h>
#include <numpy/arrayobject.h>
#include <pthread.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "thread_args.h"
#include "conv1d_operations.h"
#include "double_array_operations.h"
#include "float_array_operations.h"



/*!
 * # doubleConv1dPrep_
 *
 * Performs key steps for orthogonal random features based convolution
 * with multithreading using SORF. It is assumed that caller
 * has checked dimensions and they are all correct.
 *
 * ## Args:
 *
 * + `radem` The stacks of diagonal matrices used in
 * the transform. Must be of shape (3 x 1 x m * C) where m is
 * an integer that indicates the number of times we must repeat
 * the operation to generate the requested number of sampled frequencies.
 * + `reshapedX` Pointer to the first element of the array that will
 * be used for the convolution. A copy of this array is modified
 * rather than the original. Shape is (N x D x C). C must be
 * a power of 2.
 * + `numThreads` The number of threads to use
 * + `reshapedDim0` The first dimension of reshapedX
 * + `reshapedDim1` The second dimension of reshapedX
 * + `reshapedDim2` The last dimension of reshapedX
 * + `startPosition` Where to start when reading through radem.
 * + `numFreqs` The number of frequencies to sample.
 * numFreqs must be equal to shape[2] of radem.
 *
 * ## Returns:
 * "error" if an error, "no_error" otherwise.
 */
const char *doubleConv1dPrep_(int8_t *radem, double *reshapedX,
            int numThreads, int reshapedDim0,
            int reshapedDim1, int reshapedDim2,
            int startPosition, int numFreqs)
{

    struct ThreadConv1dDoubleArgs *th_args = malloc(numThreads * sizeof(struct ThreadConv1dDoubleArgs));
    if (th_args == NULL){
        PyErr_SetString(PyExc_ValueError, "System out of memory! RUN FOR YOUR LIFE!!!!");
        return "error";
    }
    //Note the variable length arrays, which are fine with gcc BUT may be a problem for some older
    //C++ compilers.
    int i, threadFlags[numThreads];
    int iret[numThreads];
    void *retval[numThreads];
    pthread_t thread_id[numThreads];
    int chunkSize = (reshapedDim0 + numThreads - 1) / numThreads;
    
    //We assume here that numFreqs is an integer multiple of reshapedDim2.
    //Caller must check this -- the Cython wrapper does.
    for (i=0; i < numThreads; i++){
        th_args[i].startPosition = startPosition;
        th_args[i].startRow = i * chunkSize;
        th_args[i].endRow = (i + 1) * chunkSize;
        if (th_args[i].endRow > reshapedDim0)
            th_args[i].endRow = reshapedDim0;
        th_args[i].reshapedDim1 = reshapedDim1;
        th_args[i].reshapedDim2 = reshapedDim2;
        th_args[i].numFreqs = numFreqs;
        th_args[i].rademArray = radem;
        th_args[i].reshapedXArray = reshapedX;
    }

    for (i=0; i < numThreads; i++){
        iret[i] = pthread_create(&thread_id[i], NULL, doubleThreadConv1d, &th_args[i]);
        if (iret[i]){
            PyErr_SetString(PyExc_ValueError, "fastHadamardTransform failed to create a thread!");
            return "error";
        }
    }
    for (i=0; i < numThreads; i++)
        threadFlags[i] = pthread_join(thread_id[i], &retval[i]);
    
    return "no_error";

}


/*!
 * # floatConv1dPrep_
 *
 * Performs key steps for orthogonal random features based convolution
 * with multithreading, when input is a single-precision array.
 * It is assumed that caller has checked dimensions and they are all correct.
 *
 * ## Args:
 *
 * + `radem` The stacks of diagonal matrices used in
 * the transform. Must be of shape (3 x 1 x m * C) where m is
 * an integer that indicates the number of times we must repeat
 * the operation to generate the requested number of sampled frequencies.
 * + `reshapedX` Pointer to the first element of the array that will
 * be used for the convolution. A copy of this array is modified
 * rather than the original. Shape is (N x D x C). C must be
 * a power of 2.
 * + `numThreads` The number of threads to use
 * + `reshapedDim0` The first dimension of reshapedX
 * + `reshapedDim1` The second dimension of reshapedX
 * + `reshapedDim2` The last dimension of reshapedX
 * + `startPosition` Where to start when reading through radem.
 * + `numFreqs` The number of frequencies to sample.
 * numFreqs must be equal to shape[2] of radem.
 *
 * ## Returns:
 * "error" if an error, "no_error" otherwise.
 */
const char *floatConv1dPrep_(int8_t *radem, float *reshapedX,
            int numThreads, int reshapedDim0,
            int reshapedDim1, int reshapedDim2,
            int startPosition, int numFreqs)
{

    struct ThreadConv1dFloatArgs *th_args = malloc(numThreads * sizeof(struct ThreadConv1dFloatArgs));
    if (th_args == NULL){
        PyErr_SetString(PyExc_ValueError, "System out of memory! RUN FOR YOUR LIFE!!!!");
        return "error";
    }
    //Note the variable length arrays, which are fine with gcc BUT may be a problem for some older
    //C++ compilers.
    int i, threadFlags[numThreads];
    int iret[numThreads];
    void *retval[numThreads];
    pthread_t thread_id[numThreads];
    int chunkSize = (reshapedDim0 + numThreads - 1) / numThreads;
    
    //We assume here that numFreqs is an integer multiple of reshapedDim2.
    //Caller must check this -- the Cython wrapper does.
    for (i=0; i < numThreads; i++){
        th_args[i].startPosition = startPosition;
        th_args[i].startRow = i * chunkSize;
        th_args[i].endRow = (i + 1) * chunkSize;
        if (th_args[i].endRow > reshapedDim0)
            th_args[i].endRow = reshapedDim0;
        th_args[i].reshapedDim1 = reshapedDim1;
        th_args[i].reshapedDim2 = reshapedDim2;
        th_args[i].numFreqs = numFreqs;
        th_args[i].rademArray = radem;
        th_args[i].reshapedXArray = reshapedX;
    }
    for (i=0; i < numThreads; i++){
        iret[i] = pthread_create(&thread_id[i], NULL, floatThreadConv1d, &th_args[i]);
        if (iret[i]){
            PyErr_SetString(PyExc_ValueError, "fastHadamardTransform failed to create a thread!");
            return "error";
        }
    }
    for (i=0; i < numThreads; i++)
        threadFlags[i] = pthread_join(thread_id[i], &retval[i]);
    
    return "no_error";

}



/*!
 * # doubleThreadConv1d
 *
 * Performs orthogonal random features based convolution
 * for a single thread. reshapedX is split up into
 * num_threads chunks each with a start and end row. This
 * function works on double-precision arrays only.
 *
 * ## Args:
 * + `sharedArgs` A void pointer to a ThreadConv1dDoubleArgs struct
 * containing pointers to the arrays needed to execute the
 * transform, the start and end rows etc.
 */
void *doubleThreadConv1d(void *sharedArgs){
    struct ThreadConv1dDoubleArgs *thArgs = (struct ThreadConv1dDoubleArgs *)sharedArgs;
    
    doubleConv1dMultiplyByRadem(thArgs->reshapedXArray,
                    thArgs->rademArray, thArgs->startRow,
                    thArgs->endRow, thArgs->reshapedDim1, 
                    thArgs->reshapedDim2, thArgs->startPosition);
    doubleTransformRows3D(thArgs->reshapedXArray, thArgs->startRow,
                    thArgs->endRow, thArgs->reshapedDim1, 
                    thArgs->reshapedDim2);
    doubleConv1dMultiplyByRadem(thArgs->reshapedXArray,
                    thArgs->rademArray + thArgs->numFreqs,
                    thArgs->startRow, thArgs->endRow,
                    thArgs->reshapedDim1, thArgs->reshapedDim2,
                    thArgs->startPosition);
    doubleTransformRows3D(thArgs->reshapedXArray, thArgs->startRow,
                    thArgs->endRow, thArgs->reshapedDim1, 
                    thArgs->reshapedDim2);
    
    doubleConv1dMultiplyByRadem(thArgs->reshapedXArray,
                    thArgs->rademArray + 2 * thArgs->numFreqs,
                    thArgs->startRow, thArgs->endRow,
                    thArgs->reshapedDim1, thArgs->reshapedDim2,
                    thArgs->startPosition);
    doubleTransformRows3D(thArgs->reshapedXArray, thArgs->startRow,
                    thArgs->endRow, thArgs->reshapedDim1, 
                    thArgs->reshapedDim2);
    return NULL;
}




/*!
 * # floatThreadConv1d
 *
 * Performs orthogonal random features based convolution
 * for a single thread. reshapedX is split up into
 * num_threads chunks each with a start and end row. This
 * function works on single-precision arrays only.
 *
 * ## Args:
 * + `sharedArgs` A void pointer to a ThreadConv1dFloatArgs struct
 * containing pointers to the arrays needed to execute the
 * transform, the start and end rows etc.
 */
void *floatThreadConv1d(void *sharedArgs){
    struct ThreadConv1dFloatArgs *thArgs = (struct ThreadConv1dFloatArgs *)sharedArgs;
    
    floatConv1dMultiplyByRadem(thArgs->reshapedXArray,
                    thArgs->rademArray, thArgs->startRow,
                    thArgs->endRow, thArgs->reshapedDim1, 
                    thArgs->reshapedDim2, thArgs->startPosition);
    floatTransformRows3D(thArgs->reshapedXArray, thArgs->startRow,
                    thArgs->endRow, thArgs->reshapedDim1, 
                    thArgs->reshapedDim2);
    floatConv1dMultiplyByRadem(thArgs->reshapedXArray,
                    thArgs->rademArray + thArgs->numFreqs,
                    thArgs->startRow, thArgs->endRow,
                    thArgs->reshapedDim1, thArgs->reshapedDim2,
                    thArgs->startPosition);
    floatTransformRows3D(thArgs->reshapedXArray, thArgs->startRow,
                    thArgs->endRow, thArgs->reshapedDim1, 
                    thArgs->reshapedDim2);
    
    floatConv1dMultiplyByRadem(thArgs->reshapedXArray,
                    thArgs->rademArray + 2 * thArgs->numFreqs,
                    thArgs->startRow, thArgs->endRow,
                    thArgs->reshapedDim1, thArgs->reshapedDim2,
                    thArgs->startPosition);
    floatTransformRows3D(thArgs->reshapedXArray, thArgs->startRow,
                    thArgs->endRow, thArgs->reshapedDim1, 
                    thArgs->reshapedDim2);
    return NULL;
}
