/*!
 * # conv1d_operations.c
 *
 * This module performs operations unique to the convolution
 * kernels in xGPR, essentially orthogonal random features based
 * convolution, for non-RBF kernels.
 */
#include <Python.h>
#include <vector>
#include <thread>
#include <math.h>
#include "conv1d_operations.h"
#include "../shared_fht_functions/hadamard_transforms.h"
#include "../shared_fht_functions/shared_rfgen_ops.h"




/*!
 * # conv1dMaxpoolFeatureGen_
 *
 * Generates random features for a maxpool (rather than RBF) type pre-kernel.
 *
 * ## Args:
 *
 * + `radem` The stacks of diagonal matrices used in
 * the transform. Must be of shape (3 x 1 x m * C) where m is
 * an integer that indicates the number of times we must repeat
 * the operation to generate the requested number of sampled frequencies.
 * + `xdata` The raw input 3d array, of shape (N x D x K).
 * + `chiArr` The diagonal array by which to multiply the SORF products.
 * + `outputArray` The double-precision array in which the results are
 * stored.
 * + `seqlengths` The length of each sequence in the input. Of shape (N).
 * + `dim0` The first dimension of xdata
 * + `dim1` The second dimension of xdata
 * + `dim2` The last dimension of xdata
 * + `numThreads` The number of threads to use
 * + `numFreqs` The number of frequencies to sample.
 * numFreqs must be equal <= shape[2] of radem.
 * + `convWidth` The width of the convolution to perform.
 * + `paddedBufferSize` dim2 of the copy buffer to create to perform
 * the convolution.
 * + `simplex` If True, apply the simplex modification of Reid et al. 2023.
 *
 * ## Returns:
 * "error" if an error, "no_error" otherwise.
 */
template <typename T>
const char *conv1dMaxpoolFeatureGen_(int8_t *radem, T xdata[],
            T chiArr[], double *outputArray, int32_t *seqlengths,
            int xdim0, int xdim1, int xdim2, int numThreads, int numFreqs,
            int convWidth, int paddedBufferSize,
            bool simplex){

    if (numThreads > xdim0)
        numThreads = xdim0;

    std::vector<std::thread> threads(numThreads);
    int startRow, endRow;
    int chunkSize = (xdim0 + numThreads - 1) / numThreads;
    
    for (int i=0; i < numThreads; i++){
        startRow = i * chunkSize;
        endRow = (i + 1) * chunkSize;
        if (endRow > xdim0)
            endRow = xdim0;

        if (!simplex){
            threads[i] = std::thread(&allInOneConvMaxpoolGen<T>, xdata,
                radem, chiArr, outputArray, seqlengths, xdim1, xdim2,
                numFreqs, startRow, endRow, convWidth, paddedBufferSize);
        }
        else{
            threads[i] = std::thread(&allInOneConvMaxpoolSimplex<T>, xdata,
                radem, chiArr, outputArray, seqlengths, xdim1, xdim2,
                numFreqs, startRow, endRow, convWidth, paddedBufferSize);
        }
    }

    for (auto& th : threads)
        th.join();

    return "no_error";
}
template const char *conv1dMaxpoolFeatureGen_<float>(int8_t *radem, float xdata[],
            float chiArr[], double *outputArray, int32_t *seqlengths,
            int xdim0, int xdim1, int xdim2, int numThreads, int numFreqs,
            int convWidth, int paddedBufferSize, bool simplex);
template const char *conv1dMaxpoolFeatureGen_<double>(int8_t *radem, double xdata[],
            double chiArr[], double *outputArray, int32_t *seqlengths,
            int xdim0, int xdim1, int xdim2, int numThreads, int numFreqs,
            int convWidth, int paddedBufferSize, bool simplex);



/*!
 * # allInOneConvMaxpoolGen
 *
 * Performs the maxpool-based convolution kernel feature generation
 * process for the input, for one thread.
 */
template <typename T>
void *allInOneConvMaxpoolGen(T xdata[], int8_t *rademArray, T chiArr[],
        double *outputArray, int32_t *seqlengths, int dim1, int dim2,
        int numFreqs, int startRow, int endRow,
        int convWidth, int paddedBufferSize){

    int numKmers;
    int32_t seqlength;
    int numRepeats = (numFreqs + paddedBufferSize - 1) / paddedBufferSize;
    //Notice that we don't have error handling here...very naughty. Out of
    //memory should be extremely rare since we are only allocating memory
    //for one row of the convolution. TODO: add error handling here.
    T *copyBuffer = new T[paddedBufferSize];
    T *xElement;

    for (int i=startRow; i < endRow; i++){
        seqlength = seqlengths[i];
        numKmers = seqlength - convWidth + 1;

        for (int j=0; j < numKmers; j++){
            int repeatPosition = 0;
            xElement = xdata + i * dim1 * dim2 + j * dim2;

            for (int k=0; k < numRepeats; k++){
                for (int m=0; m < (convWidth * dim2); m++)
                    copyBuffer[m] = xElement[m];
                for (int m=(convWidth * dim2); m < paddedBufferSize; m++)
                    copyBuffer[m] = 0;

                singleVectorSORF(copyBuffer, rademArray, repeatPosition,
                        numFreqs, paddedBufferSize);
                singleVectorMaxpoolPostProcess(copyBuffer, chiArr, outputArray,
                        paddedBufferSize, numFreqs, i, k);
                repeatPosition += paddedBufferSize;
            }
        }
    }
    delete[] copyBuffer;

    return NULL;
}



/*!
 * # allInOneConvMaxpoolSimplex
 *
 * Performs the maxpool-based convolution kernel feature generation
 * process for the input, for one thread, using the simplex
 * modification.
 */
template <typename T>
void *allInOneConvMaxpoolSimplex(T xdata[], int8_t *rademArray, T chiArr[],
        double *outputArray, int32_t *seqlengths, int dim1, int dim2,
        int numFreqs, int startRow, int endRow,
        int convWidth, int paddedBufferSize){

    int numKmers;
    int32_t seqlength;
    int numRepeats = (numFreqs + paddedBufferSize - 1) / paddedBufferSize;
    //Notice that we don't have error handling here...very naughty. Out of
    //memory should be extremely rare since we are only allocating memory
    //for one row of the convolution. TODO: add error handling here.
    T *copyBuffer = new T[paddedBufferSize];
    T *xElement;

    for (int i=startRow; i < endRow; i++){
        seqlength = seqlengths[i];
        numKmers = seqlength - convWidth + 1;

        for (int j=0; j < numKmers; j++){
            int repeatPosition = 0;
            xElement = xdata + i * dim1 * dim2 + j * dim2;

            for (int k=0; k < numRepeats; k++){
                for (int m=0; m < (convWidth * dim2); m++)
                    copyBuffer[m] = xElement[m];
                for (int m=(convWidth * dim2); m < paddedBufferSize; m++)
                    copyBuffer[m] = 0;

                singleVectorSORFSimplex(copyBuffer, rademArray, repeatPosition,
                        numFreqs, paddedBufferSize);
                singleVectorMaxpoolPostProcess(copyBuffer, chiArr, outputArray,
                        paddedBufferSize, numFreqs, i, k);
                repeatPosition += paddedBufferSize;
            }
        }
    }
    delete[] copyBuffer;

    return NULL;
}





/*!
 * # singleVectorMaxpoolPostProcess
 *
 * Performs the last steps in RBF-based convolution kernel feature
 * generation for a single convolution element.
 *
 * ## Args:
 * + `xdata` Pointer to the first element of the array that has been
 * used for the convolution. Shape is (C). C must be a power of 2.
 * + `chiArr` Pointer to the first element of chiArr, a diagonal matrix
 * that will be multipled against xdata.
 * + `outputArray` A pointer to the first element of the array in which
 * the output will be stored.
 * + `dim2` The last dimension of xdata
 * + `numFreqs` The number of frequencies to sample.
 * + `rowNumber` The row of the output array to use.
 * + `repeatNum` The repeat number
 * + `convWidth` The convolution width
 *
 */
template <typename T>
void singleVectorMaxpoolPostProcess(const T xdata[],
        const T chiArr[], double *outputArray,
        int dim2, int numFreqs,
        int rowNumber, int repeatNum){

    int outputStart = repeatNum * dim2;
    T prodVal;
    double *__restrict xOut;
    const T *chiIn;
    //NOTE: MIN is defined in the header.
    int endPosition = MIN(numFreqs, (repeatNum + 1) * dim2);
    endPosition -= outputStart;

    chiIn = chiArr + outputStart;
    xOut = outputArray + outputStart + rowNumber * numFreqs;

    #pragma omp simd
    for (int i=0; i < endPosition; i++){
        prodVal = xdata[i] * chiIn[i];
        //NOTE: MAX is defined in the header.
        *xOut = MAX(*xOut, prodVal);
        xOut++;
    }
}
