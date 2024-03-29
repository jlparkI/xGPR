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
 * + `scalingTerm` The scaling term to apply for the random feature generation.
 * + `scalingType` An int that is one of 0, 1 or 2 to indicate what type of
 * additional scaling (if any) to perform.
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
            int convWidth, int paddedBufferSize,
            double scalingTerm, int scalingType){
    if (numThreads > dim0)
        numThreads = dim0;

    std::vector<std::thread> threads(numThreads);
    int startRow, endRow;
    int chunkSize = (dim0 + numThreads - 1) / numThreads;
    
    for (int i=0; i < numThreads; i++){
        startRow = i * chunkSize;
        endRow = (i + 1) * chunkSize;
        if (endRow > dim0)
            endRow = dim0;

        threads[i] = std::thread(&allInOneConvRBFGen<T>, xdata,
                radem, chiArr, outputArray,
                seqlengths, dim1, dim2, numFreqs, rademShape2,
                startRow, endRow, convWidth, paddedBufferSize,
                scalingTerm, scalingType);
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
 * + `scalingTerm` The scaling term to apply for the random feature generation.
 * + `scalingType` An int that is one of 0, 1 or 2 to indicate what type of
 * additional scaling (if any) to perform.
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
            int paddedBufferSize,
            double scalingTerm, int scalingType){
    if (numThreads > dim0)
        numThreads = dim0;

    std::vector<std::thread> threads(numThreads);
    int startRow, endRow;
    int chunkSize = (dim0 + numThreads - 1) / numThreads;
    
    for (int i=0; i < numThreads; i++){
        startRow = i * chunkSize;
        endRow = (i + 1) * chunkSize;
        if (endRow > dim0)
            endRow = dim0;
        threads[i] = std::thread(&allInOneConvRBFGrad<T>, xdata,
                radem, chiArr, outputArray, seqlengths, gradientArray, dim1,
                dim2, numFreqs, rademShape2, startRow,
                endRow, convWidth, paddedBufferSize,
                scalingTerm, scalingType, sigma);
    }

    for (auto& th : threads)
        th.join();

    return "no_error";
}



/*!
 * # allInOneConvRBFGen
 *
 * Performs the RBF-based convolution kernel feature generation
 * process for the input, for one thread.
 */
template <typename T>
void *allInOneConvRBFGen(T xdata[], int8_t *rademArray, T chiArr[],
        double *outputArray, int32_t *seqlengths, int dim1, int dim2,
        int numFreqs, int rademShape2, int startRow, int endRow,
        int convWidth, int paddedBufferSize,
        double scalingTerm, int scalingType){

    int numKmers;
    int32_t seqlength;
    int numRepeats = (numFreqs + paddedBufferSize - 1) / paddedBufferSize;
    double rowScaler;
    //Notice that we don't have error handling here...very naughty. Out of
    //memory should be extremely rare since we are only allocating memory
    //for one row of the convolution. TODO: add error handling here.
    T *copyBuffer = new T[paddedBufferSize];
    T *xElement;

    for (int i=startRow; i < endRow; i++){
        seqlength = seqlengths[i];
        numKmers = seqlength - convWidth + 1;
        if (scalingType == 1)
            rowScaler = scalingTerm / sqrt( (double)numKmers);
        else if (scalingType == 2)
            rowScaler = scalingTerm / (double)numKmers;
        else
            rowScaler = scalingTerm;

        for (int j=0; j < numKmers; j++){
            int repeatPosition = 0;
            xElement = xdata + i * dim1 * dim2 + j * dim2;

            for (int k=0; k < numRepeats; k++){
                for (int m=0; m < (convWidth * dim2); m++)
                    copyBuffer[m] = xElement[m];
                for (int m=(convWidth * dim2); m < paddedBufferSize; m++)
                    copyBuffer[m] = 0;

                singleVectorSORF(copyBuffer, rademArray, repeatPosition,
                        rademShape2, paddedBufferSize);
                singleVectorRBFPostProcess(copyBuffer, chiArr, outputArray,
                        paddedBufferSize, numFreqs, i, k, rowScaler);
                repeatPosition += paddedBufferSize;
            }
        }
    }
    delete[] copyBuffer;

    return NULL;
}


/*!
 * # allInOneConvRBFGrad
 *
 * Performs the RBF-based convolution kernel feature generation
 * process for the input, for one thread, and calculates the
 * gradient, which is stored in a separate array.
 */
template <typename T>
void *allInOneConvRBFGrad(T xdata[], int8_t *rademArray, T chiArr[],
        double *outputArray, int32_t *seqlengths, double *gradientArray,
        int dim1, int dim2, int numFreqs, int rademShape2, int startRow,
        int endRow, int convWidth, int paddedBufferSize,
        double scalingTerm, int scalingType, T sigma){

    int numKmers;
    int32_t seqlength;
    int numRepeats = (numFreqs + paddedBufferSize - 1) / paddedBufferSize;
    double rowScaler;
    //Notice that we don't have error handling here...very naughty. Out of
    //memory should be extremely rare since we are only allocating memory
    //for one row of the convolution. TODO: add error handling here.
    T *copyBuffer = new T[paddedBufferSize];
    T *xElement;

    for (int i=startRow; i < endRow; i++){
        seqlength = seqlengths[i];
        numKmers = seqlength - convWidth + 1;
        if (scalingType == 1)
            rowScaler = scalingTerm / sqrt( (double)numKmers);
        else if (scalingType == 2)
            rowScaler = scalingTerm / (double)numKmers;
        else
            rowScaler = scalingTerm;

        for (int j=0; j < numKmers; j++){
            int repeatPosition = 0;
            xElement = xdata + i * dim1 * dim2 + j * dim2;

            for (int k=0; k < numRepeats; k++){
                for (int m=0; m < (convWidth * dim2); m++)
                    copyBuffer[m] = xElement[m];
                for (int m=(convWidth * dim2); m < paddedBufferSize; m++)
                    copyBuffer[m] = 0;

                singleVectorSORF(copyBuffer, rademArray, repeatPosition,
                        rademShape2, paddedBufferSize);
                singleVectorRBFPostGrad(copyBuffer, chiArr, outputArray,
                        gradientArray, sigma, paddedBufferSize, numFreqs,
                        i, k, rowScaler);
                repeatPosition += paddedBufferSize;
            }
        }
    }
    delete[] copyBuffer;

    return NULL;
}





/*!
 * # singleVectorRBFPostProcess
 *
 * Performs the last steps in RBF-based convolution kernel feature
 * generation for a single convolution element.
 *
 * ## Args:
 * + `xdata` Pointer to the first element of the array that has been
 * used for the convolution. Shape is (C). C must be a power of 2.
 * + `chiArr` Pointer to the first element of chiArr, a diagonal array
 * that will be multipled against xdata.
 * + `outputArray` A pointer to the first element of the array in which
 * the output will be stored.
 * + `dim2` The last dimension of xdata
 * + `numFreqs` The number of frequencies to sample.
 * + `rowNumber` The row of the output array to use.
 * + `repeatNum` The repeat number
 * + `convWidth` The convolution width
 * + `scalingTerm` The scaling term to apply for the random feature generation.
 *
 */
template <typename T>
void singleVectorRBFPostProcess(const T xdata[],
        const T chiArr[], double *outputArray,
        int dim2, int numFreqs,
        int rowNumber, int repeatNum,
        double scalingTerm){

    int outputStart = repeatNum * dim2;
    T prodVal;
    double *__restrict xOut;
    const T *chiIn;
    //NOTE: MIN is defined in the header.
    int endPosition = MIN(numFreqs, (repeatNum + 1) * dim2);
    endPosition -= outputStart;

    chiIn = chiArr + outputStart;
    xOut = outputArray + 2 * outputStart + rowNumber * 2 * numFreqs;

    for (int i=0; i < endPosition; i++){
        prodVal = xdata[i] * chiIn[i];
        *xOut += cos(prodVal) * scalingTerm;
        xOut++;
        *xOut += sin(prodVal) * scalingTerm;
        xOut++;
    }
}



/*!
 * # singleVectorRBFPostGrad
 *
 * Performs the last steps in RBF-based convolution kernel feature
 * generation for a single convolution element.
 *
 * ## Args:
 * + `xdata` Pointer to the first element of the array that has been
 * used for the convolution. Shape is (C). C must be a power of 2.
 * + `chiArr` Pointer to the first element of chiArr, a diagonal array
 * that will be multipled against xdata.
 * + `outputArray` A pointer to the first element of the array in which
 * the output will be stored.
 * + `gradientArray` A pointer to the first element of the array in
 * which the gradient will be stored.
 * + `sigma` The sigma hyperparameter.
 * + `dim2` The last dimension of xdata
 * + `numFreqs` The number of frequencies to sample.
 * + `rowNumber` The row of the output array to use.
 * + `repeatNum` The repeat number
 * + `convWidth` The convolution width
 * + `scalingTerm` The scaling term to apply for the random feature generation.
 *
 */
template <typename T>
void singleVectorRBFPostGrad(const T xdata[],
        const T chiArr[], double *outputArray,
        double *gradientArray, T sigma,
        int dim2, int numFreqs,
        int rowNumber, int repeatNum,
        double scalingTerm){

    int outputStart = repeatNum * dim2;
    T prodVal, gradVal, cosVal, sinVal;
    double *__restrict xOut, *__restrict gradOut;
    const T *chiIn;
    //NOTE: MIN is defined in the header.
    int endPosition = MIN(numFreqs, (repeatNum + 1) * dim2);
    endPosition -= outputStart;

    chiIn = chiArr + outputStart;
    xOut = outputArray + 2 * outputStart + rowNumber * 2 * numFreqs;
    gradOut = gradientArray + 2 * outputStart + rowNumber * 2 * numFreqs;

    for (int i=0; i < endPosition; i++){
        gradVal = xdata[i] * chiIn[i];
        prodVal = gradVal * sigma;
        cosVal = cos(prodVal) * scalingTerm;
        sinVal = sin(prodVal) * scalingTerm;
        *xOut += cosVal;
        xOut++;
        *xOut += sinVal;
        xOut++;
        *gradOut -= sinVal * gradVal;
        gradOut++;
        *gradOut += cosVal * gradVal;
        gradOut++;
    }
}


//Instantiate the templates the wrapper will need to access.
template const char *convRBFFeatureGen_<float>(int8_t *radem, float xdata[],
            float chiArr[], double *outputArray, int32_t *seqlengths,
            int numThreads, int dim0, int dim1, int dim2, int numFreqs,
            int rademShape2, int convWidth, int paddedBufferSize,
            double scalingTerm, int scalingType);
template const char *convRBFFeatureGen_<double>(int8_t *radem, double xdata[],
            double chiArr[], double *outputArray, int32_t *seqlengths,
            int numThreads, int dim0, int dim1, int dim2, int numFreqs,
            int rademShape2, int convWidth, int paddedBufferSize,
            double scalingTerm, int scalingType);

template const char *convRBFGrad_<float>(int8_t *radem, float xdata[],
            float chiArr[], double *outputArray, int32_t *seqlengths,
            double *gradientArray, float sigma,
            int numThreads, int dim0, int dim1, int dim2,
            int numFreqs, int rademShape2,
            int convWidth, int paddedBufferSize,
            double scalingTerm, int scalingType);
template const char *convRBFGrad_<double>(int8_t *radem, double xdata[],
            double chiArr[], double *outputArray, int32_t *seqlengths,
            double *gradientArray, double sigma,
            int numThreads, int dim0, int dim1, int dim2,
            int numFreqs, int rademShape2,
            int convWidth, int paddedBufferSize,
            double scalingTerm, int scalingType);
