/*!
 * # arccos_convolution.cpp
 *
 * This module performs operations unique to the ArcCos-based convolution
 * kernels in xGPR. It contains the following functions:
 *
 * + convArcCosFeatureGen_
 * Performs all steps required to generate random features for ArcCos-based
 * convolution kernels.
 *
 * + threadConvArcCosGen
 * Called once for each thread when generating ArcCos-based convolution
 * features.
 *
 * + arcCosGenPostProcessOrder1
 * Performs the last step in ArcCos-based convolution feature generation
 * if the kernel is order 1.
 *
 * + arcCosGenPostProcessOrder2
 * Performs the last step in ArcCos-based convolution feature generation
 * if the kernel is order 2.
 */
#include <Python.h>
#include <vector>
#include <thread>
#include <math.h>
#include "arccos_convolution.h"
#include "../shared_fht_functions/basic_array_operations.h"






/*!
 * # convArcCosFeatureGen_
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
template <typename T>
const char *convArcCosFeatureGen_(int8_t *radem, T reshapedX[],
            T copyBuffer[], T chiArr[], double *outputArray,
            int numThreads, int reshapedDim0,
            int reshapedDim1, int reshapedDim2,
            int numFreqs, int rademShape2,
            int kernelOrder)
{
    if (numThreads > reshapedDim0)
        numThreads = reshapedDim0;

    std::vector<std::thread> threads(numThreads);
    int startRow, endRow;
    int chunkSize = (reshapedDim0 + numThreads - 1) / numThreads;
    
    for (int i=0; i < numThreads; i++){
        startRow = i * chunkSize;
        endRow = (i + 1) * chunkSize;
        if (endRow > reshapedDim0)
            endRow = reshapedDim0;
        threads[i] = std::thread(&threadConvArcCosGen<T>, reshapedX,
                copyBuffer, chiArr, radem, outputArray,
                startRow, endRow, reshapedDim1, reshapedDim2,
                rademShape2, numFreqs, kernelOrder);
    }

    for (auto& th : threads)
        th.join();
    return "no_error";
}






/*!
 * # threadConvArcCosGen
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
template <typename T>
void *threadConvArcCosGen(T reshapedXArray[], T copyBuffer[], T chiArr[],
        int8_t *rademArray, double *outputArray, int startRow,
        int endRow, int reshapedDim1, int reshapedDim2,
        int rademShape2, int numFreqs, int kernelOrder){
    int i, numRepeats, startPosition = 0;
    numRepeats = (numFreqs +
            reshapedDim2 - 1) / reshapedDim2;

    for (i=0; i < numRepeats; i++){
        conv1dRademAndCopy<T>(reshapedXArray,
                    copyBuffer,
                    rademArray, startRow,
                    endRow, reshapedDim1, 
                    reshapedDim2, startPosition);
        transformRows3D<T>(copyBuffer, startRow,
                    endRow, reshapedDim1, 
                    reshapedDim2);
        conv1dMultiplyByRadem<T>(copyBuffer,
                    rademArray + rademShape2,
                    startRow, endRow,
                    reshapedDim1, reshapedDim2,
                    startPosition);
        transformRows3D<T>(copyBuffer, startRow,
                    endRow, reshapedDim1, 
                    reshapedDim2);
    
        conv1dMultiplyByRadem<T>(copyBuffer,
                    rademArray + 2 * rademShape2,
                    startRow, endRow,
                    reshapedDim1, reshapedDim2,
                    startPosition);
        transformRows3D(copyBuffer, startRow,
                    endRow, reshapedDim1, 
                    reshapedDim2);
        if (kernelOrder == 1){
            arcCosPostProcessOrder1<T>(copyBuffer, chiArr,
                outputArray, reshapedDim1,
                reshapedDim2, numFreqs, startRow,
                endRow, i);
        }
        else{
            arcCosPostProcessOrder2<T>(copyBuffer, chiArr,
                outputArray, reshapedDim1,
                reshapedDim2, numFreqs, startRow,
                endRow, i);
        }
        
        startPosition += reshapedDim2;
    }
    return NULL;
}







/*!
 * # arcCosPostProcessOrder1
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
template <typename T>
void arcCosPostProcessOrder1(T reshapedX[], T chiArr[],
        double *outputArray, int reshapedDim1,
        int reshapedDim2, int numFreqs,
        int startRow, int endRow, int repeatNum){
    int i, j, k, outputStart;
    double *xOut;
    T *xIn, *chiIn;
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
 * # arcCosPostProcessOrder2
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
template <typename T>
void arcCosPostProcessOrder2(T reshapedX[], T chiArr[],
        double *outputArray, int reshapedDim1,
        int reshapedDim2, int numFreqs,
        int startRow, int endRow, int repeatNum){
    int i, j, k, outputStart;
    T prodVal;
    double *xOut;
    T *xIn, *chiIn;
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

//Explicitly instantiate function wrapper will use.
template const char *convArcCosFeatureGen_<float>(int8_t *radem, float reshapedX[],
            float copyBuffer[], float chiArr[], double *outputArray,
            int numThreads, int reshapedDim0,
            int reshapedDim1, int reshapedDim2,
            int numFreqs, int rademShape2,
            int kernelOrder);
template const char *convArcCosFeatureGen_<double>(int8_t *radem, double reshapedX[],
            double copyBuffer[], double chiArr[], double *outputArray,
            int numThreads, int reshapedDim0,
            int reshapedDim1, int reshapedDim2,
            int numFreqs, int rademShape2,
            int kernelOrder);
