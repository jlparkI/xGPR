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
 * + doubleConvRBFFeatureGen_
 * Performs all steps required to generate random features for RBF-based
 * convolution kernels (FHTConv1d, GraphConv1d, ARD variants of GraphConv1d).
 *
 * + floatConvRBFFeatureGen_
 * Performs all steps required to generate random features for RBF-based
 * convolution kernels (FHTConv1d, GraphConv1d, ARD variants of GraphConv1d).
 *
 * + doubleConvRBFGrad_
 * Performs all steps required to generate random features for RBF-based
 * convolution kernels (FHTConv1d, GraphConv1d, ARD variants of GraphConv1d)
 * and additionally calculate the gradient w/r/t lengthscale.
 *
 * + floatConvRBFGrad_
 * Performs all steps required to generate random features for RBF-based
 * convolution kernels (FHTConv1d, GraphConv1d, ARD variants of GraphConv1d)
 * and additionally calculate the gradient w/r/t lengthscale.
 * 
 * + FloatThreadConv1d
 * Called once by doubleConv1dPrep_ for each thread.
 *
 * + DoubleThreadConv1d
 * Called once by doubleConv1dPrep_ for each thread.
 *
 * + FloatThreadConvRBFGen
 * Called once for each thread when generating RBF-based convolution
 * features.
 *
 * + DoubleThreadConvRBFGen
 * Called once for each thread when generating RBF-based convolution
 * features.
 *
 * + FloatThreadConvRBFGrad
 * Called once for each thread when generating RBF-based convolution
 * features with additional gradient calculation.
 *
 * + DoubleThreadConvRBFGrad
 * Called once for each thread when generating RBF-based convolution
 * features with additional gradient calculation.
 * 
 * + floatRBFGenPostProcess
 * Performs the last step in RBF-based convolution feature generation.
 *
 * + doubleRBFGenPostProcess
 * Performs the last step in RBF-based convolution feature generation.
 *
 * + floatRBFGenGrad
 * Performs the last step in RBF-based convolution feature generation
 * and gradient calculation.
 *
 * + doubleRBFGenGrad
 * Performs the last step in RBF-based convolution feature generation
 * and gradient calculation.
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
#include "conv1d_operations.h"
#include "../double_array_operations.h"
#include "../float_array_operations.h"



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
 * be used for the convolution. Shape is (N x D x C). C must be
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
    if (numThreads > reshapedDim0)
        numThreads = reshapedDim0;

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
 * be used for the convolution. Shape is (N x D x C). C must be
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

    if (numThreads > reshapedDim0)
        numThreads = reshapedDim0;

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
 * # doubleConvRBFFeatureGen_
 * Performs all steps required to generate random features for RBF-based
 * convolution kernels (FHTConv1d, GraphConv1d, ARD variants of GraphConv1d).
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
 * + `copyBuffer` An array of the same size and shape as reshapedX into
 * which reshapedX can be copied. copyBuffer can then be modified in place
 * to generate the random features.
 * + `chiArr` A diagonal array that will be multiplied against the output
 * of the SORF operation. Must be of shape m * C.
 * + `outputArray` The output array. Must be of shape N x 2 * m * C.
 * + `numThreads` The number of threads to use
 * + `reshapedDim0` The first dimension of reshapedX
 * + `reshapedDim1` The second dimension of reshapedX
 * + `reshapedDim2` The last dimension of reshapedX
 * + `numFreqs` The number of frequencies to sample.
 * numFreqs must be equal to shape[2] of radem (m * C).
 *
 * ## Returns:
 * "error" if an error, "no_error" otherwise.
 */
const char *doubleConvRBFFeatureGen_(int8_t *radem, double *reshapedX,
            double *copyBuffer, double *chiArr, double *outputArray,
            int numThreads, int reshapedDim0,
            int reshapedDim1, int reshapedDim2,
            int numFreqs)
{
    if (numThreads > reshapedDim0)
        numThreads = reshapedDim0;

    struct ThreadConvRBFDoubleArgs *th_args = malloc(numThreads * sizeof(struct ThreadConvRBFDoubleArgs));
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
        th_args[i].startRow = i * chunkSize;
        th_args[i].endRow = (i + 1) * chunkSize;
        if (th_args[i].endRow > reshapedDim0)
            th_args[i].endRow = reshapedDim0;
        th_args[i].reshapedDim1 = reshapedDim1;
        th_args[i].reshapedDim2 = reshapedDim2;
        th_args[i].numFreqs = numFreqs;
        th_args[i].rademArray = radem;
        th_args[i].reshapedXArray = reshapedX;
        th_args[i].chiArr = chiArr;
        th_args[i].copyBuffer = copyBuffer;
        th_args[i].outputArray = outputArray;
    }

    for (i=0; i < numThreads; i++){
        iret[i] = pthread_create(&thread_id[i], NULL, doubleThreadConvRBFGen, &th_args[i]);
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
 * # floatConvRBFFeatureGen_
 * Performs all steps required to generate random features for RBF-based
 * convolution kernels (FHTConv1d, GraphConv1d, ARD variants of GraphConv1d).
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
 * + `copyBuffer` An array of the same size and shape as reshapedX into
 * which reshapedX can be copied. copyBuffer can then be modified in place
 * to generate the random features.
 * + `chiArr` A diagonal array that will be multiplied against the output
 * of the SORF operation. Must be of shape m * C.
 * + `outputArray` The output array. Must be of shape N x 2 * m * C.
 * + `numThreads` The number of threads to use
 * + `reshapedDim0` The first dimension of reshapedX
 * + `reshapedDim1` The second dimension of reshapedX
 * + `reshapedDim2` The last dimension of reshapedX
 * + `numFreqs` The number of frequencies to sample.
 * numFreqs must be equal to shape[2] of radem (m * C).
 *
 * ## Returns:
 * "error" if an error, "no_error" otherwise.
 */
const char *floatConvRBFFeatureGen_(int8_t *radem, float *reshapedX,
            float *copyBuffer, float *chiArr, double *outputArray,
            int numThreads, int reshapedDim0,
            int reshapedDim1, int reshapedDim2,
            int numFreqs)
{
    if (numThreads > reshapedDim0)
        numThreads = reshapedDim0;

    struct ThreadConvRBFFloatArgs *th_args = malloc(numThreads * sizeof(struct ThreadConvRBFFloatArgs));
    if (th_args == NULL){
        PyErr_SetString(PyExc_ValueError, "System out of memory!");
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
        th_args[i].startRow = i * chunkSize;
        th_args[i].endRow = (i + 1) * chunkSize;
        if (th_args[i].endRow > reshapedDim0)
            th_args[i].endRow = reshapedDim0;
        th_args[i].reshapedDim1 = reshapedDim1;
        th_args[i].reshapedDim2 = reshapedDim2;
        th_args[i].numFreqs = numFreqs;
        th_args[i].rademArray = radem;
        th_args[i].reshapedXArray = reshapedX;
        th_args[i].chiArr = chiArr;
        th_args[i].copyBuffer = copyBuffer;
        th_args[i].outputArray = outputArray;
    }

    for (i=0; i < numThreads; i++){
        iret[i] = pthread_create(&thread_id[i], NULL, floatThreadConvRBFGen, &th_args[i]);
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
 * # doubleConvRBFGrad_
 * Performs all steps required to generate random features for RBF-based
 * convolution kernels (FHTConv1d, GraphConv1d, ARD variants of GraphConv1d),
 * TOGETHER with the gradient.
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
 * + `copyBuffer` An array of the same size and shape as reshapedX into
 * which reshapedX can be copied. copyBuffer can then be modified in place
 * to generate the random features.
 * + `chiArr` A diagonal array that will be multiplied against the output
 * of the SORF operation. Must be of shape m * C.
 * + `outputArray` The output array. Must be of shape N x 2 * m * C.
 * + `gradientArray` The array in which the gradient will be stored.
 * + `numThreads` The number of threads to use
 * + `reshapedDim0` The first dimension of reshapedX
 * + `reshapedDim1` The second dimension of reshapedX
 * + `reshapedDim2` The last dimension of reshapedX
 * + `numFreqs` The number of frequencies to sample.
 * numFreqs must be equal to shape[2] of radem (m * C).
 *
 * ## Returns:
 * "error" if an error, "no_error" otherwise.
 */
const char *doubleConvRBFGrad_(int8_t *radem, double *reshapedX,
            double *copyBuffer, double *chiArr, double *outputArray,
            double *gradientArray, double sigma,
            int numThreads, int reshapedDim0,
            int reshapedDim1, int reshapedDim2,
            int numFreqs)
{
    if (numThreads > reshapedDim0)
        numThreads = reshapedDim0;

    struct ThreadConvRBFDoubleArgs *th_args = malloc(numThreads * sizeof(struct ThreadConvRBFDoubleArgs));
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
        th_args[i].startRow = i * chunkSize;
        th_args[i].endRow = (i + 1) * chunkSize;
        if (th_args[i].endRow > reshapedDim0)
            th_args[i].endRow = reshapedDim0;
        th_args[i].reshapedDim1 = reshapedDim1;
        th_args[i].reshapedDim2 = reshapedDim2;
        th_args[i].numFreqs = numFreqs;
        th_args[i].rademArray = radem;
        th_args[i].reshapedXArray = reshapedX;
        th_args[i].chiArr = chiArr;
        th_args[i].copyBuffer = copyBuffer;
        th_args[i].outputArray = outputArray;
        th_args[i].gradientArray = gradientArray;
        th_args[i].sigma = sigma;
    }

    for (i=0; i < numThreads; i++){
        iret[i] = pthread_create(&thread_id[i], NULL, doubleThreadConvRBFGrad, &th_args[i]);
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
 * # floatConvRBFGrad_
 * Performs all steps required to generate random features for RBF-based
 * convolution kernels (FHTConv1d, GraphConv1d, ARD variants of GraphConv1d),
 * TOGETHER with the gradient.
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
 * + `copyBuffer` An array of the same size and shape as reshapedX into
 * which reshapedX can be copied. copyBuffer can then be modified in place
 * to generate the random features.
 * + `chiArr` A diagonal array that will be multiplied against the output
 * of the SORF operation. Must be of shape m * C.
 * + `outputArray` The output array. Must be of shape N x 2 * m * C.
 * + `gradientArray` The array in which the gradient will be stored.
 * + `numThreads` The number of threads to use
 * + `reshapedDim0` The first dimension of reshapedX
 * + `reshapedDim1` The second dimension of reshapedX
 * + `reshapedDim2` The last dimension of reshapedX
 * + `numFreqs` The number of frequencies to sample.
 * numFreqs must be equal to shape[2] of radem (m * C).
 *
 * ## Returns:
 * "error" if an error, "no_error" otherwise.
 */
const char *floatConvRBFGrad_(int8_t *radem, float *reshapedX,
            float *copyBuffer, float *chiArr, double *outputArray,
            double *gradientArray, float sigma,
            int numThreads, int reshapedDim0,
            int reshapedDim1, int reshapedDim2,
            int numFreqs)
{
    if (numThreads > reshapedDim0)
        numThreads = reshapedDim0;

    struct ThreadConvRBFFloatArgs *th_args = malloc(numThreads * sizeof(struct ThreadConvRBFFloatArgs));
    if (th_args == NULL){
        PyErr_SetString(PyExc_ValueError, "System out of memory!");
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
        th_args[i].startRow = i * chunkSize;
        th_args[i].endRow = (i + 1) * chunkSize;
        if (th_args[i].endRow > reshapedDim0)
            th_args[i].endRow = reshapedDim0;
        th_args[i].reshapedDim1 = reshapedDim1;
        th_args[i].reshapedDim2 = reshapedDim2;
        th_args[i].numFreqs = numFreqs;
        th_args[i].rademArray = radem;
        th_args[i].reshapedXArray = reshapedX;
        th_args[i].chiArr = chiArr;
        th_args[i].copyBuffer = copyBuffer;
        th_args[i].outputArray = outputArray;
        th_args[i].gradientArray = gradientArray;
        th_args[i].sigma = sigma;
    }

    for (i=0; i < numThreads; i++){
        iret[i] = pthread_create(&thread_id[i], NULL, floatThreadConvRBFGrad, &th_args[i]);
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
 * # doubleThreadConv1d
 *
 * Performs orthogonal random features based convolution
 * for a single thread. reshapedX is split up into
 * num_threads chunks each with a start and end row.
 *
 * ## Args:
 * + `sharedArgs` A void pointer to a struct
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
 * num_threads chunks each with a start and end row.
 *
 * ## Args:
 * + `sharedArgs` A void pointer to a struct
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



/*!
 * # doubleThreadConvRBFGen
 *
 * Performs the RBF-based convolution kernel feature generation
 * process for the input, for one thread. reshapedX is split up into
 * num_threads chunks each with a start and end row.
 *
 * ## Args:
 * + `sharedArgs` A void pointer to a struct
 * containing pointers to the arrays needed to execute the
 * transform, the start and end rows etc.
 */
void *doubleThreadConvRBFGen(void *sharedArgs){
    struct ThreadConvRBFDoubleArgs *thArgs = (struct ThreadConvRBFDoubleArgs *)sharedArgs;
    int i, numRepeats, startPosition = 0;
    numRepeats = (thArgs->numFreqs +
            thArgs->reshapedDim2 - 1) / thArgs->reshapedDim2;

    for (i=0; i < numRepeats; i++){
        doubleConv1dRademAndCopy(thArgs->reshapedXArray,
                    thArgs->copyBuffer,
                    thArgs->rademArray, thArgs->startRow,
                    thArgs->endRow, thArgs->reshapedDim1, 
                    thArgs->reshapedDim2, startPosition);
        doubleTransformRows3D(thArgs->copyBuffer, thArgs->startRow,
                    thArgs->endRow, thArgs->reshapedDim1, 
                    thArgs->reshapedDim2);
        doubleConv1dMultiplyByRadem(thArgs->copyBuffer,
                    thArgs->rademArray + thArgs->numFreqs,
                    thArgs->startRow, thArgs->endRow,
                    thArgs->reshapedDim1, thArgs->reshapedDim2,
                    startPosition);
        doubleTransformRows3D(thArgs->copyBuffer, thArgs->startRow,
                    thArgs->endRow, thArgs->reshapedDim1, 
                    thArgs->reshapedDim2);
    
        doubleConv1dMultiplyByRadem(thArgs->copyBuffer,
                    thArgs->rademArray + 2 * thArgs->numFreqs,
                    thArgs->startRow, thArgs->endRow,
                    thArgs->reshapedDim1, thArgs->reshapedDim2,
                    startPosition);
        doubleTransformRows3D(thArgs->copyBuffer, thArgs->startRow,
                    thArgs->endRow, thArgs->reshapedDim1, 
                    thArgs->reshapedDim2);
        doubleRBFPostProcess(thArgs->copyBuffer, thArgs->chiArr,
            thArgs->outputArray, thArgs->reshapedDim1,
            thArgs->reshapedDim2, thArgs->numFreqs, thArgs->startRow,
            thArgs->endRow, i);

        startPosition += thArgs->reshapedDim2;
    }
    return NULL;
}


/*!
 * # floatThreadConvRBFGen
 *
 * Performs the RBF-based convolution kernel feature generation
 * process for the input, for one thread. reshapedX is split up into
 * num_threads chunks each with a start and end row.
 *
 * ## Args:
 * + `sharedArgs` A void pointer to a struct
 * containing pointers to the arrays needed to execute the
 * transform, the start and end rows etc.
 */
void *floatThreadConvRBFGen(void *sharedArgs){
    struct ThreadConvRBFFloatArgs *thArgs = (struct ThreadConvRBFFloatArgs *)sharedArgs;
    int i, numRepeats, startPosition = 0;
    numRepeats = (thArgs->numFreqs +
            thArgs->reshapedDim2 - 1) / thArgs->reshapedDim2;

    for (i=0; i < numRepeats; i++){
        floatConv1dRademAndCopy(thArgs->reshapedXArray,
                    thArgs->copyBuffer,
                    thArgs->rademArray, thArgs->startRow,
                    thArgs->endRow, thArgs->reshapedDim1, 
                    thArgs->reshapedDim2, startPosition);
        floatTransformRows3D(thArgs->copyBuffer, thArgs->startRow,
                    thArgs->endRow, thArgs->reshapedDim1, 
                    thArgs->reshapedDim2);
        floatConv1dMultiplyByRadem(thArgs->copyBuffer,
                    thArgs->rademArray + thArgs->numFreqs,
                    thArgs->startRow, thArgs->endRow,
                    thArgs->reshapedDim1, thArgs->reshapedDim2,
                    startPosition);
        floatTransformRows3D(thArgs->copyBuffer, thArgs->startRow,
                    thArgs->endRow, thArgs->reshapedDim1, 
                    thArgs->reshapedDim2);
    
        floatConv1dMultiplyByRadem(thArgs->copyBuffer,
                    thArgs->rademArray + 2 * thArgs->numFreqs,
                    thArgs->startRow, thArgs->endRow,
                    thArgs->reshapedDim1, thArgs->reshapedDim2,
                    startPosition);
        floatTransformRows3D(thArgs->copyBuffer, thArgs->startRow,
                    thArgs->endRow, thArgs->reshapedDim1, 
                    thArgs->reshapedDim2);
        floatRBFPostProcess(thArgs->copyBuffer, thArgs->chiArr,
            thArgs->outputArray, thArgs->reshapedDim1,
            thArgs->reshapedDim2, thArgs->numFreqs, thArgs->startRow,
            thArgs->endRow, i);
        
        startPosition += thArgs->reshapedDim2;
    }
    return NULL;
}


/*!
 * # doubleThreadConvRBFGrad
 *
 * Performs the RBF-based convolution kernel feature generation
 * process for the input, for one thread. reshapedX is split up into
 * num_threads chunks each with a start and end row.
 *
 * ## Args:
 * + `sharedArgs` A void pointer to a struct
 * containing pointers to the arrays needed to execute the
 * transform, the start and end rows etc.
 */
void *doubleThreadConvRBFGrad(void *sharedArgs){
    struct ThreadConvRBFDoubleArgs *thArgs = (struct ThreadConvRBFDoubleArgs *)sharedArgs;
    int i, numRepeats, startPosition = 0;
    numRepeats = (thArgs->numFreqs +
            thArgs->reshapedDim2 - 1) / thArgs->reshapedDim2;

    for (i=0; i < numRepeats; i++){
        doubleConv1dRademAndCopy(thArgs->reshapedXArray,
                    thArgs->copyBuffer,
                    thArgs->rademArray, thArgs->startRow,
                    thArgs->endRow, thArgs->reshapedDim1, 
                    thArgs->reshapedDim2, startPosition);
        doubleTransformRows3D(thArgs->copyBuffer, thArgs->startRow,
                    thArgs->endRow, thArgs->reshapedDim1, 
                    thArgs->reshapedDim2);
        doubleConv1dMultiplyByRadem(thArgs->copyBuffer,
                    thArgs->rademArray + thArgs->numFreqs,
                    thArgs->startRow, thArgs->endRow,
                    thArgs->reshapedDim1, thArgs->reshapedDim2,
                    startPosition);
        doubleTransformRows3D(thArgs->copyBuffer, thArgs->startRow,
                    thArgs->endRow, thArgs->reshapedDim1, 
                    thArgs->reshapedDim2);
    
        doubleConv1dMultiplyByRadem(thArgs->copyBuffer,
                    thArgs->rademArray + 2 * thArgs->numFreqs,
                    thArgs->startRow, thArgs->endRow,
                    thArgs->reshapedDim1, thArgs->reshapedDim2,
                    startPosition);
        doubleTransformRows3D(thArgs->copyBuffer, thArgs->startRow,
                    thArgs->endRow, thArgs->reshapedDim1, 
                    thArgs->reshapedDim2);
        doubleRBFPostGrad(thArgs->copyBuffer, thArgs->chiArr,
            thArgs->outputArray, thArgs->gradientArray, thArgs->reshapedDim1,
            thArgs->reshapedDim2, thArgs->numFreqs, thArgs->startRow, 
            thArgs->endRow, i, thArgs->sigma);

        startPosition += thArgs->reshapedDim2;
    }
    return NULL;
}


/*!
 * # floatThreadConvRBFGrad
 *
 * Performs the RBF-based convolution kernel feature generation
 * process for the input, for one thread. reshapedX is split up into
 * num_threads chunks each with a start and end row.
 *
 * ## Args:
 * + `sharedArgs` A void pointer to a struct
 * containing pointers to the arrays needed to execute the
 * transform, the start and end rows etc.
 */
void *floatThreadConvRBFGrad(void *sharedArgs){
    struct ThreadConvRBFFloatArgs *thArgs = (struct ThreadConvRBFFloatArgs *)sharedArgs;
    int i, numRepeats, startPosition = 0;
    numRepeats = (thArgs->numFreqs +
            thArgs->reshapedDim2 - 1) / thArgs->reshapedDim2;

    for (i=0; i < numRepeats; i++){
        floatConv1dRademAndCopy(thArgs->reshapedXArray,
                    thArgs->copyBuffer,
                    thArgs->rademArray, thArgs->startRow,
                    thArgs->endRow, thArgs->reshapedDim1, 
                    thArgs->reshapedDim2, startPosition);
        floatTransformRows3D(thArgs->copyBuffer, thArgs->startRow,
                    thArgs->endRow, thArgs->reshapedDim1, 
                    thArgs->reshapedDim2);
        floatConv1dMultiplyByRadem(thArgs->copyBuffer,
                    thArgs->rademArray + thArgs->numFreqs,
                    thArgs->startRow, thArgs->endRow,
                    thArgs->reshapedDim1, thArgs->reshapedDim2,
                    startPosition);
        floatTransformRows3D(thArgs->copyBuffer, thArgs->startRow,
                    thArgs->endRow, thArgs->reshapedDim1, 
                    thArgs->reshapedDim2);
    
        floatConv1dMultiplyByRadem(thArgs->copyBuffer,
                    thArgs->rademArray + 2 * thArgs->numFreqs,
                    thArgs->startRow, thArgs->endRow,
                    thArgs->reshapedDim1, thArgs->reshapedDim2,
                    startPosition);
        floatTransformRows3D(thArgs->copyBuffer, thArgs->startRow,
                    thArgs->endRow, thArgs->reshapedDim1, 
                    thArgs->reshapedDim2);
        floatRBFPostGrad(thArgs->copyBuffer, thArgs->chiArr,
            thArgs->outputArray, thArgs->gradientArray, thArgs->reshapedDim1,
            thArgs->reshapedDim2, thArgs->numFreqs, thArgs->startRow,
            thArgs->endRow, i, thArgs->sigma);
        
        startPosition += thArgs->reshapedDim2;
    }
    return NULL;
}

/*!
 * # doubleRBFPostProcess
 *
 * Performs the last steps in RBF-based convolution kernel feature
 * generation.
 *
 * ## Args:
 * + `reshapedX` Pointer to the first element of the array that has been
 * used for the convolution. Shape is (N x D x C). C must be
 * a power of 2.
 * + `chiArr` Pointer to the first element of chiArr, a diagonal array
 * that will be multipled against reshapedX.
 * + `outputArray` A pointer to the first element of the array in which
 * the output will be stored.
 * + `reshapedDim0` The first dimension of reshapedX
 * + `reshapedDim1` The second dimension of reshapedX
 * + `reshapedDim2` The last dimension of reshapedX
 * + `numFreqs` The number of frequencies to sample.
 * + `startRow` The first row of the input to work on
 * + `endRow` The last row of the input to work on
 * + `repeatNum` The repeat number
 */
void doubleRBFPostProcess(double *reshapedX, double *chiArr,
        double *outputArray, int reshapedDim1,
        int reshapedDim2, int numFreqs,
        int startRow, int endRow, int repeatNum){
    int i, j, k, lenOutputRow, chiStart, outputStart;
    double prodVal;
    double *xIn, *xOut, *chiIn;

    lenOutputRow = 2 * numFreqs;
    chiStart = repeatNum * reshapedDim2;
    outputStart = 2 * repeatNum * reshapedDim2;

    xIn = reshapedX + startRow * reshapedDim1 * reshapedDim2;

    for (i=startRow; i < endRow; i++){
        xOut = outputArray + i * lenOutputRow + outputStart;
        chiIn = chiArr + chiStart;

        for (k=0; k < reshapedDim1; k++){
            for (j=0; j < reshapedDim2; j++){
                prodVal = *xIn * chiIn[j];
                xOut[j] += cos(prodVal);
                xOut[j+reshapedDim2] += sin(prodVal);
                xIn++;
            }
        }
    }
}




/*!
 * # floatRBFPostProcess
 *
 * Performs the last steps in RBF-based convolution kernel feature
 * generation.
 *
 * ## Args:
 * + `reshapedX` Pointer to the first element of the array that has been
 * used for the convolution. Shape is (N x D x C). C must be
 * a power of 2.
 * + `chiArr` Pointer to the first element of chiArr, a diagonal array
 * that will be multipled against reshapedX.
 * + `outputArray` A pointer to the first element of the array in which
 * the output will be stored.
 * + `reshapedDim0` The first dimension of reshapedX
 * + `reshapedDim1` The second dimension of reshapedX
 * + `reshapedDim2` The last dimension of reshapedX
 * + `numFreqs` The number of frequencies to sample.
 * + `startRow` The first row of the input to work on
 * + `endRow` The last row of the input to work on
 * + `repeatNum` The repeat number
 */
void floatRBFPostProcess(float *reshapedX, float *chiArr,
        double *outputArray, int reshapedDim1,
        int reshapedDim2, int numFreqs,
        int startRow, int endRow, int repeatNum){
    int i, j, k, lenOutputRow, outputStart, chiStart;
    float prodVal;
    double *xOut;
    float *xIn, *chiIn;

    lenOutputRow = 2 * numFreqs;
    chiStart = repeatNum * reshapedDim2;
    outputStart = 2 * repeatNum * reshapedDim2;

    xIn = reshapedX + startRow * reshapedDim1 * reshapedDim2;

    for (i=startRow; i < endRow; i++){
        xOut = outputArray + i * lenOutputRow + outputStart;
        chiIn = chiArr + chiStart;

        for (k=0; k < reshapedDim1; k++){
            for (j=0; j < reshapedDim2; j++){
                prodVal = *xIn * chiIn[j];
                xOut[j] += cosf(prodVal);
                xOut[j+reshapedDim2] += sinf(prodVal);
                xIn++;
            }
        }
    }
}




/*!
 * # doubleRBFPostGrad
 *
 * Performs the last steps in RBF-based convolution kernel feature
 * generation, while additionally calculating the gradient w/r/t
 * the lengthscale.
 *
 * ## Args:
 * + `reshapedX` Pointer to the first element of the array that has been
 * used for the convolution. Shape is (N x D x C). C must be
 * a power of 2.
 * + `chiArr` Pointer to the first element of chiArr, a diagonal array
 * that will be multipled against reshapedX.
 * + `outputArray` A pointer to the first element of the array in which
 * the output will be stored.
 * + `gradientArray` A pointer to the first element of the array in which
 * the gradient will be stored.
 * + `reshapedDim0` The first dimension of reshapedX
 * + `reshapedDim1` The second dimension of reshapedX
 * + `reshapedDim2` The last dimension of reshapedX
 * + `numFreqs` The number of frequencies to sample.
 * + `startRow` The first row of the input to work on
 * + `endRow` The last row of the input to work on
 * + `repeatNum` The repeat number
 * + `sigma` The lengthscale
 */
void doubleRBFPostGrad(double *reshapedX, double *chiArr,
        double *outputArray, double *gradientArray,
        int reshapedDim1, int reshapedDim2,
        int numFreqs, int startRow, int endRow,
        int repeatNum, double sigma){
    int i, j, k, lenOutputRow, outputStart, chiStart;
    double prodVal, gradVal, cosVal, sinVal;
    double *xIn, *xOut, *chiIn, *gradOut;

    lenOutputRow = 2 * numFreqs;
    chiStart = repeatNum * reshapedDim2;
    outputStart = 2 * repeatNum * reshapedDim2;

    xIn = reshapedX + startRow * reshapedDim1 * reshapedDim2;

    for (i=startRow; i < endRow; i++){
        xOut = outputArray + i * lenOutputRow + outputStart;
        chiIn = chiArr + chiStart;

        gradOut = gradientArray + i * lenOutputRow + outputStart;

        for (k=0; k < reshapedDim1; k++){
            for (j=0; j < reshapedDim2; j++){
                gradVal = *xIn * chiIn[j];
                prodVal = gradVal * sigma;
                cosVal = cos(prodVal);
                sinVal = sin(prodVal);
                xOut[j] += cosVal;
                xOut[j+reshapedDim2] += sinVal;
                gradOut[j] += -sinVal * gradVal;
                gradOut[j+reshapedDim2] += cosVal * gradVal;
                xIn++;
            }
        }
    }
}




/*!
 * # floatRBFPostGrad
 *
 * Performs the last steps in RBF-based convolution kernel feature
 * generation, while additionally calculating the gradient w/r/t
 * the lengthscale.
 *
 * ## Args:
 * + `reshapedX` Pointer to the first element of the array that has been
 * used for the convolution. Shape is (N x D x C). C must be
 * a power of 2.
 * + `chiArr` Pointer to the first element of chiArr, a diagonal array
 * that will be multipled against reshapedX.
 * + `outputArray` A pointer to the first element of the array in which
 * the output will be stored.
 * + `reshapedDim0` The first dimension of reshapedX
 * + `reshapedDim1` The second dimension of reshapedX
 * + `reshapedDim2` The last dimension of reshapedX
 * + `numFreqs` The number of frequencies to sample.
 * + `startRow` The first row of the input to work on
 * + `endRow` The last row of the input to work on
 * + `repeatNum` The repeat number
 * + `sigma` The lengthscale
 */
void floatRBFPostGrad(float *reshapedX, float *chiArr,
        double *outputArray, double *gradientArray,
        int reshapedDim1, int reshapedDim2,
        int numFreqs, int startRow, int endRow,
        int repeatNum, float sigma){
    int i, j, k, lenOutputRow, outputStart, chiStart;
    float prodVal, gradVal, cosVal, sinVal;
    double *xOut, *gradOut;
    float *xIn, *chiIn;

    lenOutputRow = 2 * numFreqs;
    chiStart = repeatNum * reshapedDim2;
    outputStart = 2 * repeatNum * reshapedDim2;

    xIn = reshapedX + startRow * reshapedDim1 * reshapedDim2;

    for (i=startRow; i < endRow; i++){
        xOut = outputArray + i * lenOutputRow + outputStart;
        gradOut = gradientArray + i * lenOutputRow + outputStart;
        chiIn = chiArr + chiStart;

        for (k=0; k < reshapedDim1; k++){
            for (j=0; j < reshapedDim2; j++){
                gradVal = *xIn * chiIn[j];
                prodVal = gradVal * sigma;
                cosVal = cosf(prodVal);
                sinVal = sinf(prodVal);
                xOut[j] += cosVal;
                xOut[j+reshapedDim2] += sinVal;
                gradOut[j] += -sinVal * gradVal;
                gradOut[j+reshapedDim2] += cosVal * gradVal;
                xIn++;
            }
        }
    }
}
