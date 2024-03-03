/*!
 * # rbf_convolution.c
 *
 * This module performs operations unique to the RBF-based convolution
 * kernels in xGPR.
 */
#include <Python.h>
#include <thread>
#include <vector>
#include <math.h>
#include "rbf_convolution.h"
#include "../shared_fht_functions/hadamard_transforms.h"
#include "../shared_fht_functions/shared_rfgen_ops.h"






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
 * + `xdata` Pointer to the first element of the array that will
 * be used for the convolution. A copy of this array is modified
 * rather than the original. Shape is (N x D x C). C must be
 * a power of 2.
 * + `chiArr` A diagonal array that will be multiplied against the output
 * of the SORF operation. Of shape numFreqs.
 * + `outputArray` The output array. Of shape (N, 2 * numFreqs).
 * + `seqlengths` The length of each sequence in the input. Of shape (N).
 * + `numThreads` The number of threads to use
 * + `dim0` The first dimension of xdata
 * + `dim1` The second dimension of xdata
 * + `dim2` The last dimension of xdata
 * + `numFreqs` The number of frequencies to sample. Must be <=
 * radem.shape[2].
 * + `rademShape2` The number of elements in one row of radem.
 * + `convWidth` The width of the convolution.
 * + `paddedBufferSize` dim2 of the copy buffer to create to perform
 * the convolution.
 *
 * ## Returns:
 * "error" if an error, "no_error" otherwise.
 */
template <typename T>
const char *convRBFFeatureGen_(int8_t *radem, T xdata[],
            T chiArr[], double *outputArray,
            int32_t *seqlengths,
            int numThreads, int dim0,
            int dim1, int dim2,
            int numFreqs, int rademShape2,
            int convWidth, int paddedBufferSize){
    if (numThreads > dim0)
        numThreads = dim0;

    int bufferRowSize = (dim1 - convWidth + 1) * paddedBufferSize * dim0;

    T *copyBuffer = new (std::nothrow) T[bufferRowSize];
    if (copyBuffer == NULL)
        return "Out of memory! Could not allocate a copy buffer. Check input sizes.";

    std::vector<std::thread> threads(numThreads);
    int startRow, endRow;
    int chunkSize = (dim0 + numThreads - 1) / numThreads;
    
    for (int i=0; i < numThreads; i++){
        startRow = i * chunkSize;
        endRow = (i + 1) * chunkSize;
        if (endRow > dim0)
            endRow = dim0;

        threads[i] = std::thread(&threadConvRBFGen<T>, xdata,
                copyBuffer, radem, chiArr, outputArray,
                seqlengths, dim1, dim2, numFreqs, rademShape2,
                startRow, endRow, convWidth, paddedBufferSize);
    }

    for (auto& th : threads)
        th.join();

    delete[] copyBuffer;
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
 * + `xdata` Pointer to the first element of the array that will
 * be used for the convolution. A copy of this array is modified
 * rather than the original. Shape is (N x D x C). C must be
 * a power of 2.
 * + `copyBuffer` An array of the same size and shape as xdata into
 * which xdata can be copied. copyBuffer can then be modified in place
 * to generate the random features.
 * + `chiArr` A diagonal array that will be multiplied against the output
 * of the SORF operation. Must be of shape numFreqs.
 * + `outputArray` The output array. Must be of shape (N, 2 * numFreqs).
 * + `seqlengths` The length of each sequence in the input. Of shape (N).
 * + `gradientArray` The array in which the gradient will be stored.
 * + `sigma` The lengthscale hyperparameter.
 * + `numThreads` The number of threads to use
 * + `dim0` The first dimension of xdata
 * + `dim1` The second dimension of xdata
 * + `dim2` The last dimension of xdata
 * + `numFreqs` The number of frequencies to sample. Must be <=
 * radem.shape[2].
 * + `rademShape2` The number of elements in one row of radem.
 * + `convWidth` The width of the convolution.
 * + `paddedBufferSize` dim2 of the copy buffer to create to perform
 * the convolution.
 *
 * ## Returns:
 * "error" if an error, "no_error" otherwise.
 */
template <typename T>
const char *convRBFGrad_(int8_t *radem, T xdata[],
            T chiArr[], double *outputArray,
            int32_t *seqlengths,
            double *gradientArray, T sigma,
            int numThreads, int dim0,
            int dim1, int dim2, int numFreqs,
            int rademShape2, int convWidth,
            int paddedBufferSize){
    if (numThreads > dim0)
        numThreads = dim0;

    int bufferRowSize = (dim1 - convWidth + 1) * paddedBufferSize * dim0;

    T *copyBuffer = new (std::nothrow) T[bufferRowSize];
    if (copyBuffer == NULL)
        return "Out of memory! Could not allocate a copy buffer. Check input sizes.";

    std::vector<std::thread> threads(numThreads);
    int startRow, endRow;
    int chunkSize = (dim0 + numThreads - 1) / numThreads;
    
    for (int i=0; i < numThreads; i++){
        startRow = i * chunkSize;
        endRow = (i + 1) * chunkSize;
        if (endRow > dim0)
            endRow = dim0;
        threads[i] = std::thread(&threadConvRBFGrad<T>, xdata, copyBuffer,
                radem, chiArr, outputArray, seqlengths, gradientArray, dim1,
                dim2, numFreqs, rademShape2, startRow,
                endRow, sigma, convWidth, paddedBufferSize);
    }

    for (auto& th : threads)
        th.join();

    delete[] copyBuffer;
    return "no_error";
}






/*!
 * # threadConvRBFGen
 *
 * Performs the RBF-based convolution kernel feature generation
 * process for the input, for one thread. xdata is split up into
 * num_threads chunks each with a start and end row.
 */
template <typename T>
void *threadConvRBFGen(T xdata[], T copyBuffer[],
        int8_t *rademArray, T chiArr[], double *outputArray,
        int32_t *seqlengths, int dim1, int dim2, int numFreqs,
        int rademShape2, int startRow, int endRow,
        int convWidth, int paddedBufferSize){

    int i, numRepeats, repeatPosition = 0;
    int numKmers = dim1 - convWidth + 1;
    numRepeats = (numFreqs + paddedBufferSize - 1) / paddedBufferSize;

    for (i=0; i < numRepeats; i++){
        convSORF3DWithCopyBuffer(xdata, copyBuffer, rademArray, repeatPosition,
                startRow, endRow, dim1, dim2,
                rademShape2, convWidth, paddedBufferSize);
        RBFPostProcess<T>(copyBuffer, chiArr,
            outputArray, numKmers,
            paddedBufferSize, numFreqs, startRow,
            endRow, i, convWidth, seqlengths);
        
        repeatPosition += paddedBufferSize;
    }
    return NULL;
}




/*!
 * # threadConvRBFGrad
 *
 * Performs the RBF-based convolution kernel feature generation
 * process for the input, for one thread. xdata is split up into
 * num_threads chunks each with a start and end row.
 */
template <typename T>
void *threadConvRBFGrad(T xdata[], T copyBuffer[],
        int8_t *rademArray, T chiArr[], double *outputArray,
        int32_t *seqlengths, double *gradientArray, int dim1,
        int dim2, int numFreqs, int rademShape2,
        int startRow, int endRow, T sigma,
        int convWidth, int paddedBufferSize){

    int i, numRepeats, repeatPosition = 0;
    int numKmers = dim1 - convWidth + 1;
    numRepeats = (numFreqs + paddedBufferSize - 1) / paddedBufferSize;

    for (i=0; i < numRepeats; i++){
        convSORF3DWithCopyBuffer(xdata, copyBuffer, rademArray, repeatPosition,
                startRow, endRow, dim1, dim2,
                rademShape2, convWidth, paddedBufferSize);
        RBFPostGrad<T>(copyBuffer, chiArr,
            outputArray, gradientArray, numKmers,
            paddedBufferSize, numFreqs, startRow,
            endRow, i, sigma, convWidth, seqlengths);
        
        repeatPosition += paddedBufferSize;
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
 * + `xdata` Pointer to the first element of the array that has been
 * used for the convolution. Shape is (N x D x C). C must be
 * a power of 2.
 * + `chiArr` Pointer to the first element of chiArr, a diagonal array
 * that will be multipled against xdata.
 * + `outputArray` A pointer to the first element of the array in which
 * the output will be stored.
 * + `dim1` The second dimension of xdata
 * + `dim2` The last dimension of xdata
 * + `numFreqs` The number of frequencies to sample.
 * + `startRow` The first row of the input to work on
 * + `endRow` The last row of the input to work on
 * + `repeatNum` The repeat number
 * + `convWidth` The convolution width
 * + `seqlengths` an N-shaped array indicating the length of each
 * sequence.
 */
template <typename T>
void RBFPostProcess(const T __restrict xdata[],
        const T chiArr[], double *__restrict outputArray,
        int dim1, int dim2, int numFreqs,
        int startRow, int endRow, int repeatNum,
        int convWidth, const int32_t *seqlengths){

    int sequenceCutoff, lenOutputRow, outputStart;
    T prodVal;
    double *__restrict xOut;
    const T *__restrict xIn;
    const T *chiIn;
    int endPosition, lenInputRow = dim1 * dim2;

    outputStart = repeatNum * dim2;

    //NOTE: MIN is defined in the header.
    endPosition = MIN(numFreqs, (repeatNum + 1) * dim2);
    endPosition -= outputStart;
    lenOutputRow = 2 * numFreqs;
    chiIn = chiArr + outputStart;

    for (int i=startRow; i < endRow; i++){
        sequenceCutoff = seqlengths[i] - convWidth + 1;
        xIn = xdata + i * lenInputRow;

        for (int k=0; k < sequenceCutoff; k++){
            xOut = outputArray + i * lenOutputRow + 2 * outputStart;

            for (int j=0; j < endPosition; j++){
                prodVal = xIn[j] * chiIn[j];
                *xOut += cos(prodVal);
                xOut++;
                *xOut += sin(prodVal);
                xOut++;
            }
            xIn += dim2;
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
 * + `xdata` Pointer to the first element of the array that has been
 * used for the convolution. Shape is (N x D x C). C must be
 * a power of 2.
 * + `chiArr` Pointer to the first element of chiArr, a diagonal array
 * that will be multipled against xdata.
 * + `outputArray` A pointer to the first element of the array in which
 * the output will be stored.
 * + `gradientArray` A pointer to the first element of the array in which
 * the gradient will be stored.
 * + `dim1` The second dimension of xdata
 * + `dim2` The last dimension of xdata
 * + `numFreqs` The number of frequencies to sample.
 * + `startRow` The first row of the input to work on
 * + `endRow` The last row of the input to work on
 * + `repeatNum` The repeat number
 * + `sigma` The lengthscale hyperparameter
 * + `convWidth` The convolution width
 * + `seqlengths` an N-shaped array indicating the length of each
 * sequence.
 */
template <typename T>
void RBFPostGrad(const T __restrict xdata[],
        const T chiArr[], double *__restrict outputArray,
        double *__restrict gradientArray,
        int dim1, int dim2,
        int numFreqs, int startRow, int endRow,
        int repeatNum, T sigma,
        int convWidth, const int32_t *seqlengths){

    int sequenceCutoff, lenOutputRow, outputStart;
    T prodVal, gradVal, cosVal, sinVal;
    double *__restrict xOut, *__restrict gradOut;
    const T *__restrict xIn;
    const T *chiIn;
    int endPosition, lenInputRow = dim1 * dim2;

    outputStart = repeatNum * dim2;

    //NOTE: MIN is defined in the header.
    endPosition = MIN(numFreqs, (repeatNum + 1) * dim2);
    endPosition -= outputStart;

    lenOutputRow = 2 * numFreqs;
    chiIn = chiArr + outputStart;

    for (int i=startRow; i < endRow; i++){
        sequenceCutoff = seqlengths[i] - convWidth + 1;
        xIn = xdata + i * lenInputRow;
        xOut = outputArray + i * lenOutputRow + 2 * outputStart;
        gradOut = gradientArray + i * lenOutputRow + 2 * outputStart;

        for (int k=0; k < sequenceCutoff; k++){

            for (int j=0; j < endPosition; j++){
                gradVal = xIn[j] * chiIn[j];
                prodVal = gradVal * sigma;
                cosVal = cos(prodVal);
                sinVal = sin(prodVal);
                xOut[2*j] += cosVal;
                xOut[2*j+1] += sinVal;
                gradOut[2*j] += -sinVal * gradVal;
                gradOut[2*j+1] += cosVal * gradVal;
            }
            xIn += dim2;
        }
    }
}


//Instantiate the templates the wrapper will need to access.
template const char *convRBFFeatureGen_<float>(int8_t *radem, float xdata[],
            float chiArr[], double *outputArray, int32_t *seqlengths,
            int numThreads, int dim0, int dim1, int dim2, int numFreqs,
            int rademShape2, int convWidth, int paddedBufferSize);
template const char *convRBFFeatureGen_<double>(int8_t *radem, double xdata[],
            double chiArr[], double *outputArray, int32_t *seqlengths,
            int numThreads, int dim0, int dim1, int dim2, int numFreqs,
            int rademShape2, int convWidth, int paddedBufferSize);

template const char *convRBFGrad_<float>(int8_t *radem, float xdata[],
            float chiArr[], double *outputArray, int32_t *seqlengths,
            double *gradientArray, float sigma,
            int numThreads, int dim0, int dim1, int dim2,
            int numFreqs, int rademShape2,
            int convWidth, int paddedBufferSize);
template const char *convRBFGrad_<double>(int8_t *radem, double xdata[],
            double chiArr[], double *outputArray, int32_t *seqlengths,
            double *gradientArray, double sigma,
            int numThreads, int dim0, int dim1, int dim2,
            int numFreqs, int rademShape2,
            int convWidth, int paddedBufferSize);
