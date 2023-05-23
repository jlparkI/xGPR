/*!
 * # arccos_convolution.c
 *
 * This module performs operations unique to the ArcCos-based convolution
 * kernels in xGPR. It contains the following functions:
 *
 * + doubleConvArcCosFeatureGen_
 * Performs all steps required to generate random features for ArcCos-based
 * convolution kernels.
 *
 * + floatConvArcCosFeatureGen_
 * Performs all steps required to generate random features for ArcCos-based
 * convolution kernels.
 * 
 * + FloatThreadConvArcCosGen
 * Called once for each thread when generating ArcCos-based convolution
 * features.
 *
 * + DoubleThreadConvArcCosGen
 * Called once for each thread when generating ArcCos-based convolution
 * features.
 *
 * + floatArcCosGenPostProcessOrder1
 * Performs the last step in ArcCos-based convolution feature generation
 * if the kernel is order 1.
 *
 * + doubleArcCosGenPostProcessOrder1
 * Performs the last step in ArcCos-based convolution feature generation
 * if the kernel is order 1.
 *
 * + floatArcCosGenPostProcessOrder2
 * Performs the last step in ArcCos-based convolution feature generation
 * if the kernel is order 2.
 *
 * + doubleArcCosGenPostProcessOrder2
 * Performs the last step in ArcCos-based convolution feature generation
 * if the kernel is order 2.
 * 
 * Functions from float_ and double_ array_operations.c are called to
 * perform the Hadamard transform and diagonal matrix multiplications.
 */
#include <Python.h>
#include <pthread.h>
#include <math.h>
#include "arccos_convolution.h"
#include "../shared_fht_functions/double_array_operations.h"
#include "../shared_fht_functions/float_array_operations.h"



/*!
 * # doubleConvArcCosFeatureGen_
 * Performs all steps required to generate random features for ArcCos-based
 * convolution.
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
 * of the SORF operation. Must be of shape numFreqs.
 * + `outputArray` The output array. Must be of shape (N, numFreqs).
 * + `numThreads` The number of threads to use
 * + `reshapedDim0` The first dimension of reshapedX
 * + `reshapedDim1` The second dimension of reshapedX
 * + `reshapedDim2` The last dimension of reshapedX
 * + `numFreqs` The number of frequencies to sample. Must be <=
 * radem.shape[2].
 * + `rademShape2` The number of elements in one row of radem.
 *
 * ## Returns:
 * "error" if an error, "no_error" otherwise.
 */
const char *doubleConvArcCosFeatureGen_(int8_t *radem, double *reshapedX,
            double *copyBuffer, double *chiArr, double *outputArray,
            int numThreads, int reshapedDim0,
            int reshapedDim1, int reshapedDim2,
            int numFreqs, int rademShape2,
            int kernelOrder)
{
    if (numThreads > reshapedDim0)
        numThreads = reshapedDim0;

    struct ThreadConvArcCosDoubleArgs *th_args = malloc(numThreads * sizeof(struct ThreadConvArcCosDoubleArgs));
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
        th_args[i].rademShape2 = rademShape2;
    }

    for (i=0; i < numThreads; i++){
        iret[i] = pthread_create(&thread_id[i], NULL, doubleThreadConvArcCosGen, &th_args[i]);
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
 * # floatConvArcCosFeatureGen_
 * Performs all steps required to generate random features for ArcCos-based
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
 * of the SORF operation. Of shape numFreqs.
 * + `outputArray` The output array. Of shape (N, numFreqs).
 * + `numThreads` The number of threads to use
 * + `reshapedDim0` The first dimension of reshapedX
 * + `reshapedDim1` The second dimension of reshapedX
 * + `reshapedDim2` The last dimension of reshapedX
 * + `numFreqs` The number of frequencies to sample. Must be <=
 * radem.shape[2].
 * + `rademShape2` The number of elements in one row of radem.
 *
 * ## Returns:
 * "error" if an error, "no_error" otherwise.
 */
const char *floatConvArcCosFeatureGen_(int8_t *radem, float *reshapedX,
            float *copyBuffer, float *chiArr, double *outputArray,
            int numThreads, int reshapedDim0,
            int reshapedDim1, int reshapedDim2,
            int numFreqs, int rademShape2,
            int kernelOrder)
{
    if (numThreads > reshapedDim0)
        numThreads = reshapedDim0;

    struct ThreadConvArcCosFloatArgs *th_args = malloc(numThreads * sizeof(struct ThreadConvArcCosFloatArgs));
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
        th_args[i].rademShape2 = rademShape2;
    }

    for (i=0; i < numThreads; i++){
        iret[i] = pthread_create(&thread_id[i], NULL, floatThreadConvArcCosGen, &th_args[i]);
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
 * # doubleThreadConvArcCosGen
 *
 * Performs the ArcCos-based convolution kernel feature generation
 * process for the input, for one thread. reshapedX is split up into
 * num_threads chunks each with a start and end row.
 *
 * ## Args:
 * + `sharedArgs` A void pointer to a struct
 * containing pointers to the arrays needed to execute the
 * transform, the start and end rows etc.
 */
void *doubleThreadConvArcCosGen(void *sharedArgs){
    struct ThreadConvArcCosDoubleArgs *thArgs = (struct ThreadConvArcCosDoubleArgs *)sharedArgs;
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
                    thArgs->rademArray + thArgs->rademShape2,
                    thArgs->startRow, thArgs->endRow,
                    thArgs->reshapedDim1, thArgs->reshapedDim2,
                    startPosition);
        doubleTransformRows3D(thArgs->copyBuffer, thArgs->startRow,
                    thArgs->endRow, thArgs->reshapedDim1, 
                    thArgs->reshapedDim2);
    
        doubleConv1dMultiplyByRadem(thArgs->copyBuffer,
                    thArgs->rademArray + 2 * thArgs->rademShape2,
                    thArgs->startRow, thArgs->endRow,
                    thArgs->reshapedDim1, thArgs->reshapedDim2,
                    startPosition);
        doubleTransformRows3D(thArgs->copyBuffer, thArgs->startRow,
                    thArgs->endRow, thArgs->reshapedDim1, 
                    thArgs->reshapedDim2);

        doubleArcCosPostProcess(thArgs->copyBuffer, thArgs->chiArr,
            thArgs->outputArray, thArgs->reshapedDim1,
            thArgs->reshapedDim2, thArgs->numFreqs, thArgs->startRow,
            thArgs->endRow, i);

        startPosition += thArgs->reshapedDim2;
    }
    return NULL;
}


/*!
 * # floatThreadConvArcCosGen
 *
 * Performs the ArcCos-based convolution kernel feature generation
 * process for the input, for one thread. reshapedX is split up into
 * num_threads chunks each with a start and end row.
 *
 * ## Args:
 * + `sharedArgs` A void pointer to a struct
 * containing pointers to the arrays needed to execute the
 * transform, the start and end rows etc.
 */
void *floatThreadConvArcCosGen(void *sharedArgs){
    struct ThreadConvArcCosFloatArgs *thArgs = (struct ThreadConvArcCosFloatArgs *)sharedArgs;
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
                    thArgs->rademArray + thArgs->rademShape2,
                    thArgs->startRow, thArgs->endRow,
                    thArgs->reshapedDim1, thArgs->reshapedDim2,
                    startPosition);
        floatTransformRows3D(thArgs->copyBuffer, thArgs->startRow,
                    thArgs->endRow, thArgs->reshapedDim1, 
                    thArgs->reshapedDim2);
    
        floatConv1dMultiplyByRadem(thArgs->copyBuffer,
                    thArgs->rademArray + 2 * thArgs->rademShape2,
                    thArgs->startRow, thArgs->endRow,
                    thArgs->reshapedDim1, thArgs->reshapedDim2,
                    startPosition);
        floatTransformRows3D(thArgs->copyBuffer, thArgs->startRow,
                    thArgs->endRow, thArgs->reshapedDim1, 
                    thArgs->reshapedDim2);
        floatArcCosPostProcess(thArgs->copyBuffer, thArgs->chiArr,
            thArgs->outputArray, thArgs->reshapedDim1,
            thArgs->reshapedDim2, thArgs->numFreqs, thArgs->startRow,
            thArgs->endRow, i);
        
        startPosition += thArgs->reshapedDim2;
    }
    return NULL;
}



/*!
 * # doubleArcCosPostProcessOrder1
 *
 * Performs the last steps in ArcCos-based convolution kernel feature
 * generation if the kernel is order 1.
 *
 * ## Args:
 * + `reshapedX` Pointer to the first element of the array that has been
 * used for the convolution. Shape is (N x D x C). C must be
 * a power of 2.
 * + `chiArr` Pointer to the first element of chiArr, a diagonal array
 * that will be multipled against reshapedX.
 * + `outputArray` A pointer to the first element of the array in which
 * the output will be stored.
 * + `reshapedDim1` The second dimension of reshapedX
 * + `reshapedDim2` The last dimension of reshapedX
 * + `numFreqs` The number of frequencies to sample.
 * + `startRow` The first row of the input to work on
 * + `endRow` The last row of the input to work on
 * + `repeatNum` The repeat number
 */
void doubleArcCosPostProcessOrder1(double *reshapedX, double *chiArr,
        double *outputArray, int reshapedDim1,
        int reshapedDim2, int numFreqs,
        int startRow, int endRow, int repeatNum){
    int i, j, k, outputStart;
    double *xIn, *xOut, *chiIn;
    int endPosition, lenInputRow = reshapedDim1 * reshapedDim2;

    outputStart = repeatNum * reshapedDim2;

    //NOTE: MIN and MAX is defined in the header.
    endPosition = MIN(numFreqs, (repeatNum + 1) * reshapedDim2);
    endPosition -= outputStart;
    chiIn = chiArr + outputStart;
    xIn = reshapedX + startRow * lenInputRow;

    for (i=startRow; i < endRow; i++){
        for (k=0; k < reshapedDim1; k++){
            xOut = outputArray + i * numFreqs + outputStart;
            for (j=0; j < endPosition; j++){
                *xOut += MAX(xIn[j], 0) * chiIn[j];
                xOut++;
            }
            xIn += reshapedDim2;
        }
    }
}




/*!
 * # floatArcCosPostProcessOrder1
 *
 * Performs the last steps in ArcCos-based convolution kernel feature
 * generation if the kernel is order 1.
 *
 * ## Args:
 * + `reshapedX` Pointer to the first element of the array that has been
 * used for the convolution. Shape is (N x D x C). C must be
 * a power of 2.
 * + `chiArr` Pointer to the first element of chiArr, a diagonal array
 * that will be multipled against reshapedX.
 * + `outputArray` A pointer to the first element of the array in which
 * the output will be stored.
 * + `reshapedDim1` The second dimension of reshapedX
 * + `reshapedDim2` The last dimension of reshapedX
 * + `numFreqs` The number of frequencies to sample.
 * + `startRow` The first row of the input to work on
 * + `endRow` The last row of the input to work on
 * + `repeatNum` The repeat number
 */
void floatArcCosPostProcessOrder1(float *reshapedX, float *chiArr,
        double *outputArray, int reshapedDim1,
        int reshapedDim2, int numFreqs,
        int startRow, int endRow, int repeatNum){
    int i, j, k, outputStart;
    double *xOut;
    float *xIn, *chiIn;
    int endPosition, lenInputRow = reshapedDim1 * reshapedDim2;

    outputStart = repeatNum * reshapedDim2;

    //NOTE: MIN and MAX is defined in the header.
    endPosition = MIN(numFreqs, (repeatNum + 1) * reshapedDim2);
    endPosition -= outputStart;

    chiIn = chiArr + outputStart;

    for (i=startRow; i < endRow; i++){
        xIn = reshapedX + i * lenInputRow;

        for (k=0; k < reshapedDim1; k++){
            xOut = outputArray + i * numFreqs + outputStart;
            for (j=0; j < endPosition; j++){
                *xOut += MAX(xIn[j], 0) * chiIn[j];
                xOut++;
            }
            xIn += reshapedDim2;
        }
    }
}



/*!
 * # doubleArcCosPostProcessOrder2
 *
 * Performs the last steps in ArcCos-based convolution kernel feature
 * generation if the kernel is order 2.
 *
 * ## Args:
 * + `reshapedX` Pointer to the first element of the array that has been
 * used for the convolution. Shape is (N x D x C). C must be
 * a power of 2.
 * + `chiArr` Pointer to the first element of chiArr, a diagonal array
 * that will be multipled against reshapedX.
 * + `outputArray` A pointer to the first element of the array in which
 * the output will be stored.
 * + `reshapedDim1` The second dimension of reshapedX
 * + `reshapedDim2` The last dimension of reshapedX
 * + `numFreqs` The number of frequencies to sample.
 * + `startRow` The first row of the input to work on
 * + `endRow` The last row of the input to work on
 * + `repeatNum` The repeat number
 */
void doubleArcCosPostProcessOrder2(double *reshapedX, double *chiArr,
        double *outputArray, int reshapedDim1,
        int reshapedDim2, int numFreqs,
        int startRow, int endRow, int repeatNum){
    int i, j, k, outputStart;
    double prodVal;
    double *xIn, *xOut, *chiIn;
    int endPosition, lenInputRow = reshapedDim1 * reshapedDim2;

    outputStart = repeatNum * reshapedDim2;

    //NOTE: MIN and MAX is defined in the header.
    endPosition = MIN(numFreqs, (repeatNum + 1) * reshapedDim2);
    endPosition -= outputStart;
    chiIn = chiArr + outputStart;
    xIn = reshapedX + startRow * lenInputRow;

    for (i=startRow; i < endRow; i++){
        for (k=0; k < reshapedDim1; k++){
            xOut = outputArray + i * numFreqs + outputStart;
            for (j=0; j < endPosition; j++){
                prodVal = MAX(xIn[j], 0) * chiIn[j];
                *xOut += prodVal * prodVal;
                xOut++;
            }
            xIn += reshapedDim2;
        }
    }
}




/*!
 * # floatArcCosPostProcessOrder2
 *
 * Performs the last steps in ArcCos-based convolution kernel feature
 * generation if the kernel is order 2.
 *
 * ## Args:
 * + `reshapedX` Pointer to the first element of the array that has been
 * used for the convolution. Shape is (N x D x C). C must be
 * a power of 2.
 * + `chiArr` Pointer to the first element of chiArr, a diagonal array
 * that will be multipled against reshapedX.
 * + `outputArray` A pointer to the first element of the array in which
 * the output will be stored.
 * + `reshapedDim1` The second dimension of reshapedX
 * + `reshapedDim2` The last dimension of reshapedX
 * + `numFreqs` The number of frequencies to sample.
 * + `startRow` The first row of the input to work on
 * + `endRow` The last row of the input to work on
 * + `repeatNum` The repeat number
 */
void floatArcCosPostProcessOrder2(float *reshapedX, float *chiArr,
        double *outputArray, int reshapedDim1,
        int reshapedDim2, int numFreqs,
        int startRow, int endRow, int repeatNum){
    int i, j, k, outputStart;
    float prodVal;
    double *xOut;
    float *xIn, *chiIn;
    int endPosition, lenInputRow = reshapedDim1 * reshapedDim2;

    outputStart = repeatNum * reshapedDim2;

    //NOTE: MIN and MAX is defined in the header.
    endPosition = MIN(numFreqs, (repeatNum + 1) * reshapedDim2);
    endPosition -= outputStart;

    chiIn = chiArr + outputStart;

    for (i=startRow; i < endRow; i++){
        xIn = reshapedX + i * lenInputRow;

        for (k=0; k < reshapedDim1; k++){
            xOut = outputArray + i * numFreqs + outputStart;
            for (j=0; j < endPosition; j++){
                prodVal = MAX(xIn[j], 0) * chiIn[j];
                *xOut += prodVal * prodVal;
                xOut++;
            }
            xIn += reshapedDim2;
        }
    }
}
