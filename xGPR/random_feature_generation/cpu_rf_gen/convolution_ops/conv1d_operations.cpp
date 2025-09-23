/* Copyright (C) 2025 Jonathan Parkinson
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
// C++ headers
#include <math.h>
#include <vector>

// Library headers

// Project headers
#include "conv1d_operations.h"
#include "../shared_fht_functions/hadamard_transforms.h"
#include "../shared_fht_functions/shared_rfgen_ops.h"




namespace CPUMaxpoolKernelCalculations {



/// @brief Generates random features for the static layer FastConv1d-type
/// feature generator.
/// @param inputArr The input features that will be used to generate the RFs.
/// @param outputArr The array in which random features will be stored.
/// @param radem The array in which the diagonal Rademacher matrices are
/// stored.
/// @param chiArr The array in which the diagonal scaling matrix is stored.
/// @param seqlengths The array storing the length of each sequence in
/// inputArr.
/// @param convWidth The width of the convolution kernel.
template <typename T>
int conv1dMaxpoolFeatureGen_(nb::ndarray<T, nb::shape<-1,-1,-1>, nb::device::cpu, nb::c_contig> inputArr,
        nb::ndarray<float, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> outputArr,
        nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cpu, nb::c_contig> radem,
        nb::ndarray<T, nb::shape<-1>, nb::device::cpu, nb::c_contig> chiArr,
        nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> seqlengths,
        int convWidth) {

    // Perform safety checks. Any exceptions thrown here are handed
    // off to Python by the Nanobind wrapper. We do not expect the
    // user to see these because the Python code will always ensure
    // inputs are correct -- these are a failsafe -- so we do not need
    // to provide detailed exception messages here.
    int xDim0 = inputArr.shape(0);
    int xDim1 = inputArr.shape(1);
    int xDim2 = inputArr.shape(2);
    size_t numRffs = outputArr.shape(1);
    size_t numFreqs = chiArr.shape(0);

    T *inputPtr = static_cast<T*>(inputArr.data());
    float *outputPtr = static_cast<float*>(outputArr.data());
    T *chiPtr = static_cast<T*>(chiArr.data());
    int8_t *rademPtr = static_cast<int8_t*>(radem.data());
    int32_t *seqlengthsPtr = static_cast<int32_t*>(seqlengths.data());

    if (inputArr.shape(0) == 0 || outputArr.shape(0) != inputArr.shape(0))
        throw std::runtime_error("no datapoints");
    if (numRffs < 2 || (numRffs & 1) != 0)
        throw std::runtime_error("last dim of output must be even number");
    if ( numFreqs != numRffs || numFreqs > radem.shape(2) )
        throw std::runtime_error("incorrect number of rffs and or freqs.");

    if (seqlengths.shape(0) != inputArr.shape(0))
        throw std::runtime_error("wrong array sizes");
    if (static_cast<int>(inputArr.shape(1)) < convWidth || convWidth <= 0)
        throw std::runtime_error("invalid conv_width");

    double expectedNFreq = static_cast<double>(convWidth * inputArr.shape(2));
    expectedNFreq = MAX(expectedNFreq, 2);
    double log2Freqs = std::log2(expectedNFreq);
    log2Freqs = std::ceil(log2Freqs);
    int paddedBufferSize = std::pow(2, log2Freqs);
    int numRepeats = (numFreqs + paddedBufferSize - 1) / paddedBufferSize;

    if (radem.shape(2) % paddedBufferSize != 0 ||
            radem.shape(2) != numRepeats * paddedBufferSize) {
        throw std::runtime_error("incorrect number of rffs and or freqs.");
    }


    int32_t minSeqLength = 2147483647, maxSeqLength = 0;
    for (size_t i=0; i < seqlengths.shape(0); i++) {
        if (seqlengths(i) > maxSeqLength)
            maxSeqLength = seqlengths(i);
        if (seqlengths(i) < minSeqLength)
            minSeqLength = seqlengths(i);
    }

    if (maxSeqLength > static_cast<int32_t>(inputArr.shape(1)) || minSeqLength < convWidth) {
        throw std::runtime_error("All sequence lengths must be >= conv width and < "
                "array size.");
    }

    #pragma omp parallel
    {
    int seqlength, numKmers;
    int rademShape2 = numRepeats * paddedBufferSize;
    // Notice that we don't have error handling here...very naughty. Out of
    // memory should be extremely rare since we are only allocating memory
    // for one row of the convolution. TODO: add error handling here.
    T *copyBuffer = new T[paddedBufferSize];

    #pragma omp for
    for (int i=0; i < xDim0; i++) {
        seqlength = seqlengthsPtr[i];
        numKmers = seqlength - convWidth + 1;

        for (int j=0; j < numKmers; j++) {
            int repeatPosition = 0;
            T *xElement = inputPtr + i * xDim1 * xDim2 + j * xDim2;

            for (int k=0; k < numRepeats; k++) {
                #pragma omp simd
                for (int m=0; m < (convWidth * xDim2); m++)
                    copyBuffer[m] = xElement[m];
                #pragma omp simd
                for (int m=(convWidth * xDim2); m < paddedBufferSize; m++)
                    copyBuffer[m] = 0;

                SharedCPURandomFeatureOps::singleVectorSORF(copyBuffer,
                        rademPtr, repeatPosition, rademShape2,
                        paddedBufferSize);
                singleVectorMaxpoolPostProcess(copyBuffer, chiPtr, outputPtr,
                        paddedBufferSize, numFreqs, i, k);
                repeatPosition += paddedBufferSize;
            }
        }
    }
    delete[] copyBuffer;
    }

    return 0;
}
template int conv1dMaxpoolFeatureGen_<double>(
nb::ndarray<double, nb::shape<-1,-1,-1>, nb::device::cpu, nb::c_contig> inputArr,
nb::ndarray<float, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> outputArr,
nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cpu, nb::c_contig> radem,
nb::ndarray<double, nb::shape<-1>, nb::device::cpu, nb::c_contig> chiArr,
nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> seqlengths,
int convWidth);

template int conv1dMaxpoolFeatureGen_<float>(
nb::ndarray<float, nb::shape<-1,-1,-1>, nb::device::cpu, nb::c_contig> inputArr,
nb::ndarray<float, nb::shape<-1, -1>, nb::device::cpu, nb::c_contig> outputArr,
nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cpu, nb::c_contig> radem,
nb::ndarray<float, nb::shape<-1>, nb::device::cpu, nb::c_contig> chiArr,
nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> seqlengths,
int convWidth);



/*!
 * # allInOneConvMaxpoolGen
 *
 * Performs the maxpool-based convolution kernel feature generation
 * process for the input, for one thread.
 */
template <typename T>
void *allInOneConvMaxpoolGen(T xdata[], int8_t *rademArray, T chiArr[],
        float *outputArray, int32_t *seqlengths, int dim1, int dim2,
        int numFreqs, int startRow, int endRow,
        int convWidth, int paddedBufferSize) {

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
        const T chiArr[], float *outputArray,
        int dim2, int numFreqs,
        int rowNumber, int repeatNum) {
    int outputStart = repeatNum * dim2;
    float *__restrict xOut;
    const T *chiIn;
    // NOTE: MIN is defined in the header.
    int endPosition = MIN(numFreqs, (repeatNum + 1) * dim2);
    endPosition -= outputStart;

    chiIn = chiArr + outputStart;
    xOut = outputArray + outputStart + rowNumber * numFreqs;

    #pragma omp simd
    for (int i=0; i < endPosition; i++) {
        T prodVal = xdata[i] * chiIn[i];
        // NOTE: MAX is defined in the header.
        *xOut = MAX(*xOut, prodVal);
        xOut++;
    }
}



}  // namespace CPUMaxpoolKernelCalculations
