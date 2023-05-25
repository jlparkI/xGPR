/*!
 * # rbf_convolution.c
 *
 * This module performs operations unique to the RBF-based convolution
 * kernels in xGPR, which includes FHTConv1d and GraphConv1d. It
 * contains the following functions:
 *
 * + convRBFFeatureGen_
 * Performs all steps required to generate random features for RBF-based
 * convolution kernels (FHTConv1d, GraphConv1d, ARD variants of GraphConv1d).
 *
 * + convRBFGrad_
 * Performs all steps required to generate random features for RBF-based
 * convolution kernels (FHTConv1d, GraphConv1d, ARD variants of GraphConv1d)
 * and additionally calculate the gradient w/r/t lengthscale.
 * 
 * + FloatThreadConvRBFGen
 * Called once for each thread when generating RBF-based convolution
 * features.
 *
 * + FloatThreadConvRBFGrad
 * Called once for each thread when generating RBF-based convolution
 * features with additional gradient calculation.
 *
 * + RBFGenPostProcess
 * Performs the last step in RBF-based convolution feature generation.
 *
 * + RBFGenGrad
 * Performs the last step in RBF-based convolution feature generation
 * and gradient calculation.
 * 
 * Functions from float_ and double_ array_operations.c are called to
 * perform the Hadamard transform and diagonal matrix multiplications.
 */
#include <Python.h>
#include <thread>
#include <vector>
#include <math.h>
#include "rbf_convolution.h"
#include "../shared_fht_functions/basic_array_operations.h"






/*!
 * # convRBFFeatureGen_
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
 * of the SORF operation. Of shape numFreqs.
 * + `outputArray` The output array. Of shape (N, 2 * numFreqs).
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
const char *convRBFFeatureGen_(int8_t *radem, T reshapedX[],
            T copyBuffer[], T chiArr[], double *outputArray,
            int numThreads, int reshapedDim0,
            int reshapedDim1, int reshapedDim2,
            int numFreqs, int rademShape2)
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
        threads[i] = std::thread(&threadConvRBFGen<T>, reshapedX,
                copyBuffer, radem, chiArr, outputArray,
                reshapedDim1, reshapedDim2, numFreqs,
                rademShape2, startRow, endRow);
    }

    for (auto& th : threads)
        th.join();
    return "no_error";
}





/*!
 * # convRBFGrad_
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
 * of the SORF operation. Must be of shape numFreqs.
 * + `outputArray` The output array. Must be of shape (N, 2 * numFreqs).
 * + `gradientArray` The array in which the gradient will be stored.
 * + `sigma` The lengthscale hyperparameter.
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
const char *convRBFGrad_(int8_t *radem, T reshapedX[],
            T copyBuffer[], T chiArr[], double *outputArray,
            double *gradientArray, T sigma,
            int numThreads, int reshapedDim0,
            int reshapedDim1, int reshapedDim2,
            int numFreqs, int rademShape2)
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
        threads[i] = std::thread(&threadConvRBFGrad<T>, reshapedX, copyBuffer,
                radem, chiArr, outputArray, gradientArray, reshapedDim1,
                reshapedDim2, numFreqs, rademShape2, startRow,
                endRow, sigma);
    }

    for (auto& th : threads)
        th.join();
    return "no_error";
}






/*!
 * # threadConvRBFGen
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
template <typename T>
void *threadConvRBFGen(T reshapedXArray[], T copyBuffer[],
        int8_t *rademArray, T chiArr[], double *outputArray,
        int reshapedDim1, int reshapedDim2, int numFreqs,
        int rademShape2, int startRow, int endRow){
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
        transformRows3D<T>(copyBuffer, startRow,
                    endRow, reshapedDim1, 
                    reshapedDim2);
        RBFPostProcess<T>(copyBuffer, chiArr,
            outputArray, reshapedDim1,
            reshapedDim2, numFreqs, startRow,
            endRow, i);
        
        startPosition += reshapedDim2;
    }
    return NULL;
}




/*!
 * # threadConvRBFGrad
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
template <typename T>
void *threadConvRBFGrad(T reshapedXArray[], T copyBuffer[],
        int8_t *rademArray, T chiArr[], double *outputArray,
        double *gradientArray, int reshapedDim1,
        int reshapedDim2, int numFreqs, int rademShape2,
        int startRow, int endRow, T sigma){
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
        transformRows3D<T>(copyBuffer, startRow,
                    endRow, reshapedDim1, 
                    reshapedDim2);
        RBFPostGrad<T>(copyBuffer, chiArr,
            outputArray, gradientArray, reshapedDim1,
            reshapedDim2, numFreqs, startRow,
            endRow, i, sigma);
        
        startPosition += reshapedDim2;
    }
    return NULL;
}

/*!
 * # RBFPostProcess
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
 * + `reshapedDim1` The second dimension of reshapedX
 * + `reshapedDim2` The last dimension of reshapedX
 * + `numFreqs` The number of frequencies to sample.
 * + `startRow` The first row of the input to work on
 * + `endRow` The last row of the input to work on
 * + `repeatNum` The repeat number
 */
template <typename T>
void RBFPostProcess(T reshapedX[], T chiArr[],
        double *outputArray, int reshapedDim1,
        int reshapedDim2, int numFreqs,
        int startRow, int endRow, int repeatNum){
    int i, j, k, lenOutputRow, outputStart;
    T prodVal;
    double *xOut;
    T *xIn, *chiIn;
    int endPosition, lenInputRow = reshapedDim1 * reshapedDim2;

    outputStart = repeatNum * reshapedDim2;

    //NOTE: MIN is defined in the header.
    endPosition = MIN(numFreqs, (repeatNum + 1) * reshapedDim2);
    endPosition -= outputStart;
    lenOutputRow = 2 * numFreqs;
    chiIn = chiArr + outputStart;
    xIn = reshapedX + startRow * lenInputRow;

    for (i=startRow; i < endRow; i++){
        for (k=0; k < reshapedDim1; k++){
            xOut = outputArray + i * lenOutputRow + 2 * outputStart;
            for (j=0; j < endPosition; j++){
                prodVal = xIn[j] * chiIn[j];
                *xOut += cos(prodVal);
                xOut++;
                *xOut += sin(prodVal);
                xOut++;
            }
            xIn += reshapedDim2;
        }
    }
}





/*!
 * # RBFPostGrad
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
 * + `reshapedDim1` The second dimension of reshapedX
 * + `reshapedDim2` The last dimension of reshapedX
 * + `numFreqs` The number of frequencies to sample.
 * + `startRow` The first row of the input to work on
 * + `endRow` The last row of the input to work on
 * + `repeatNum` The repeat number
 * + `sigma` The lengthscale hyperparameter
 */
template <typename T>
void RBFPostGrad(T reshapedX[], T chiArr[],
        double *outputArray, double *gradientArray,
        int reshapedDim1, int reshapedDim2,
        int numFreqs, int startRow, int endRow,
        int repeatNum, T sigma){
    int i, j, k, lenOutputRow, outputStart;
    T prodVal, gradVal, cosVal, sinVal;
    double *xOut, *gradOut;
    T *xIn, *chiIn;
    int endPosition, lenInputRow = reshapedDim1 * reshapedDim2;

    outputStart = repeatNum * reshapedDim2;

    //NOTE: MIN is defined in the header.
    endPosition = MIN(numFreqs, (repeatNum + 1) * reshapedDim2);
    endPosition -= outputStart;

    lenOutputRow = 2 * numFreqs;
    chiIn = chiArr + outputStart;

    for (i=startRow; i < endRow; i++){
        xIn = reshapedX + i * lenInputRow;
        xOut = outputArray + i * lenOutputRow + 2 * outputStart;
        gradOut = gradientArray + i * lenOutputRow + 2 * outputStart;

        for (k=0; k < reshapedDim1; k++){
            for (j=0; j < endPosition; j++){
                gradVal = xIn[j] * chiIn[j];
                prodVal = gradVal * sigma;
                cosVal = cos(prodVal);
                sinVal = sin(prodVal);
                xOut[2*j] += cosVal;
                xOut[2*j+1] += sinVal;
                gradOut[2*j] += -sinVal * gradVal;
                gradOut[2*j+1] += cosVal * gradVal;
            }
            xIn += reshapedDim2;
        }
    }
}


//Instantiate the templates the wrapper will need to access.
template const char *convRBFFeatureGen_<float>(int8_t *radem, float reshapedX[],
            float copyBuffer[], float chiArr[], double *outputArray,
            int numThreads, int reshapedDim0,
            int reshapedDim1, int reshapedDim2,
            int numFreqs, int rademShape2);
template const char *convRBFFeatureGen_<double>(int8_t *radem, double reshapedX[],
            double copyBuffer[], double chiArr[], double *outputArray,
            int numThreads, int reshapedDim0,
            int reshapedDim1, int reshapedDim2,
            int numFreqs, int rademShape2);

template const char *convRBFGrad_<float>(int8_t *radem, float reshapedX[],
            float copyBuffer[], float chiArr[], double *outputArray,
            double *gradientArray, float sigma,
            int numThreads, int reshapedDim0,
            int reshapedDim1, int reshapedDim2,
            int numFreqs, int rademShape2);
template const char *convRBFGrad_<double>(int8_t *radem, double reshapedX[],
            double copyBuffer[], double chiArr[], double *outputArray,
            double *gradientArray, double sigma,
            int numThreads, int reshapedDim0,
            int reshapedDim1, int reshapedDim2,
            int numFreqs, int rademShape2);
