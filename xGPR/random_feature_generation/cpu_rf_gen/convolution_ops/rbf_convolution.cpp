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
#include "rbf_convolution.h"
#include "../shared_fht_functions/hadamard_transforms.h"
#include "../shared_fht_functions/shared_rfgen_ops.h"




namespace CPURBFConvolutionKernelCalculations {



/// @brief Generates random features for RBF-based convolution kernels.
/// @param inputArr The input features that will be used to generate the RFs.
/// @param outputArr The array in which random features will be stored.
/// @param radem The array in which the diagonal Rademacher matrices are
/// stored.
/// @param chiArr The array in which the diagonal scaling matrix is stored.
/// @param seqlengths The array storing the length of each sequence in
/// inputArr.
/// @param convWidth The width of the convolution kernel.
/// @param scalingType The type of scaling to perform (i.e. how to normalize
/// for different sequence lengths).
template <typename T>
int convRBFFeatureGen_(nb::ndarray<T, nb::shape<-1, -1, -1>, nb::device::cpu, nb::c_contig> inputArr,
        nb::ndarray<double, nb::shape<-1, -1>, nb::device::cpu, nb::c_contig> outputArr,
        nb::ndarray<int8_t, nb::shape<3,1, -1>, nb::device::cpu, nb::c_contig> radem,
        nb::ndarray<T, nb::shape<-1>, nb::device::cpu, nb::c_contig> chiArr,
        nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> seqlengths,
        int convWidth, int scalingType) {

    // Perform safety checks. Any exceptions thrown here are handed off
    // to Python by the Nanobind wrapper. We do not expect the user to
    // see these because the Python code will always ensure inputs are
    // correct -- these are a failsafe -- so we do not need to provide
    // detailed exception messages here.
    int xDim0 = inputArr.shape(0);
    int xDim1 = inputArr.shape(1);
    int xDim2 = inputArr.shape(2);
    int rademShape2 = radem.shape(2);
    size_t numRffs = outputArr.shape(1);
    size_t numFreqs = chiArr.shape(0);
    double scalingTerm = std::sqrt(1.0 / static_cast<double>(numFreqs));

    T *inputPtr = static_cast<T*>(inputArr.data());
    double *outputPtr = static_cast<double*>(outputArr.data());
    T *chiPtr = static_cast<T*>(chiArr.data());
    int8_t *rademPtr = static_cast<int8_t*>(radem.data());
    int32_t *seqlengthsPtr = static_cast<int32_t*>(seqlengths.data());

    if (inputArr.shape(0) == 0 || outputArr.shape(0) != inputArr.shape(0))
        throw std::runtime_error("no datapoints");
    if (numRffs < 2 || (numRffs & 1) != 0)
        throw std::runtime_error("last dim of output must be even number");
    if ( (2 * numFreqs) != numRffs || numFreqs > radem.shape(2) )
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

    if (radem.shape(2) % paddedBufferSize != 0)
        throw std::runtime_error("incorrect number of rffs and or freqs.");


    int32_t minSeqLength = 2147483647, maxSeqLength = 0;
    for (size_t i=0; i < seqlengths.shape(0); i++) {
        if (seqlengths(i) > maxSeqLength)
            maxSeqLength = seqlengths(i);
        if (seqlengths(i) < minSeqLength)
            minSeqLength = seqlengths(i);
    }

    if (maxSeqLength > static_cast<int32_t>(inputArr.shape(1)) ||
            minSeqLength < convWidth) {
        throw std::runtime_error("All sequence lengths must "
                "be >= conv width and < array size.");
    }

    #pragma omp parallel
    {
    int numKmers;
    int32_t seqlength;
    int numRepeats = (numFreqs + paddedBufferSize - 1) / paddedBufferSize;
    double rowScaler;
    // Notice that we don't have error handling here...very naughty. Out of
    // memory should be extremely rare since we are only allocating memory
    // for one row of the convolution. TODO: add error handling here.
    T *copyBuffer = new T[paddedBufferSize];
    T *xElement;

    #pragma omp for
    for (int i=0; i < xDim0; i++) {
        seqlength = seqlengthsPtr[i];
        numKmers = seqlength - convWidth + 1;
        switch (scalingType) {
            case SQRT_CONVOLUTION_SCALING:
                rowScaler = scalingTerm /
                    std::sqrt(static_cast<double>(numKmers));
                break;
            case FULL_CONVOLUTION_SCALING:
                rowScaler = scalingTerm / static_cast<double>(numKmers);
                break;
            default:
                rowScaler = scalingTerm;
                break;
        }

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
                SharedCPURandomFeatureOps::singleVectorRBFPostProcess(
                        copyBuffer, chiPtr, outputPtr, paddedBufferSize,
                        numFreqs, i, k, rowScaler);
                repeatPosition += paddedBufferSize;
            }
        }
    }
    delete[] copyBuffer;
    }

    return 0;
}
//Instantiate the templates the wrapper will need to access.
template int convRBFFeatureGen_<double>(nb::ndarray<double, nb::shape<-1, -1, -1>, nb::device::cpu, nb::c_contig> inputArr,
nb::ndarray<double, nb::shape<-1, -1>, nb::device::cpu, nb::c_contig> outputArr,
nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cpu, nb::c_contig> radem,
nb::ndarray<double, nb::shape<-1>, nb::device::cpu, nb::c_contig> chiArr,
nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> seqlengths,
int convWidth, int scalingType);

template int convRBFFeatureGen_<float>(nb::ndarray<float, nb::shape<-1, -1, -1>, nb::device::cpu, nb::c_contig> inputArr,
nb::ndarray<double, nb::shape<-1, -1>, nb::device::cpu, nb::c_contig> outputArr,
nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cpu, nb::c_contig> radem,
nb::ndarray<float, nb::shape<-1>, nb::device::cpu, nb::c_contig> chiArr,
nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> seqlengths,
int convWidth, int scalingType);





/// @brief Generates both random features and gradient for
/// RBF-based convolution kernels.
/// @param inputArr The input features that will be used to generate the RFs.
/// @param outputArr The array in which random features will be stored.
/// @param radem The array in which the diagonal Rademacher matrices are
/// stored.
/// @param chiArr The array in which the diagonal scaling matrix is stored.
/// @param seqlengths The array storing the length of each sequence in
/// inputArr.
/// @param gradArr The array in which the gradient will be stored.
/// @param convWidth The width of the convolution kernel.
/// @param sigma The lengthscale hyperparameter.
/// @param scalingType The type of scaling to perform (i.e. how to normalize
/// for different sequence lengths).
template <typename T>
int convRBFGrad_(nb::ndarray<T, nb::shape<-1, -1, -1>, nb::device::cpu, nb::c_contig> inputArr,
        nb::ndarray<double, nb::shape<-1, -1>, nb::device::cpu, nb::c_contig> outputArr,
        nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cpu, nb::c_contig> radem,
        nb::ndarray<T, nb::shape<-1>, nb::device::cpu, nb::c_contig> chiArr,
        nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> seqlengths,
        nb::ndarray<double, nb::shape<-1, -1,1>, nb::device::cpu, nb::c_contig> gradArr,
        double sigma, int convWidth, int scalingType) {

    // Perform safety checks. Any exceptions thrown here are handed off
    // to Python by the Nanobind wrapper. We do not expect the user to
    // see these because the Python code will always ensure inputs are
    // correct -- these are a failsafe -- so we do not need to provide
    // detailed exception messages here.
    int xDim0 = inputArr.shape(0);
    int xDim1 = inputArr.shape(1);
    int xDim2 = inputArr.shape(2);
    int rademShape2 = radem.shape(2);
    size_t numRffs = outputArr.shape(1);
    size_t numFreqs = chiArr.shape(0);
    double scalingTerm = std::sqrt(1.0 / static_cast<double>(numFreqs));

    T *inputPtr = static_cast<T*>(inputArr.data());
    double *outputPtr = static_cast<double*>(outputArr.data());
    T *chiPtr = static_cast<T*>(chiArr.data());
    int8_t *rademPtr = static_cast<int8_t*>(radem.data());
    int32_t *seqlengthsPtr = static_cast<int32_t*>(seqlengths.data());
    double *gradientPtr = static_cast<double*>(gradArr.data());

    if (inputArr.shape(0) == 0 || outputArr.shape(0) != inputArr.shape(0))
        throw std::runtime_error("no datapoints");
    if (numRffs < 2 || (numRffs & 1) != 0)
        throw std::runtime_error("last dim of output must be even number");
    if ( (2 * numFreqs) != numRffs || numFreqs > radem.shape(2) )
        throw std::runtime_error("incorrect number of rffs and or freqs.");

    if (seqlengths.shape(0) != inputArr.shape(0))
        throw std::runtime_error("wrong array sizes");
    if (static_cast<int>(inputArr.shape(1)) < convWidth || convWidth <= 0)
        throw std::runtime_error("invalid conv_width");

    if (gradArr.shape(0) != outputArr.shape(0) ||
            gradArr.shape(1) != outputArr.shape(1))
        throw std::runtime_error("wrong array sizes");

    double expectedNFreq = static_cast<double>(convWidth * inputArr.shape(2));
    expectedNFreq = MAX(expectedNFreq, 2);
    double log2Freqs = std::log2(expectedNFreq);
    log2Freqs = std::ceil(log2Freqs);
    int paddedBufferSize = std::pow(2, log2Freqs);

    if (radem.shape(2) % paddedBufferSize != 0)
        throw std::runtime_error("incorrect number of rffs and or freqs.");


    int32_t minSeqLength = 2147483647, maxSeqLength = 0;
    for (size_t i=0; i < seqlengths.shape(0); i++) {
        if (seqlengths(i) > maxSeqLength)
            maxSeqLength = seqlengths(i);
        if (seqlengths(i) < minSeqLength)
            minSeqLength = seqlengths(i);
    }

    if (maxSeqLength > static_cast<int32_t>(inputArr.shape(1))
            || minSeqLength < convWidth) {
        throw std::runtime_error("All sequence lengths must "
                "be >= conv width and < array size.");
    }



    #pragma omp parallel
    {
    int numKmers;
    int32_t seqlength;
    int numRepeats = (numFreqs + paddedBufferSize - 1) / paddedBufferSize;
    double rowScaler;
    // Notice that we don't have error handling here...very naughty. Out of
    // memory should be extremely rare since we are only allocating memory
    // for one row of the convolution. TODO: add error handling here.
    T *copyBuffer = new T[paddedBufferSize];
    T *xElement;

    #pragma omp for
    for (int i=0; i < xDim0; i++) {
        seqlength = seqlengthsPtr[i];
        numKmers = seqlength - convWidth + 1;
        switch (scalingType) {
            case SQRT_CONVOLUTION_SCALING:
                rowScaler = scalingTerm / std::sqrt(
                        static_cast<double>(numKmers));
                break;
            case FULL_CONVOLUTION_SCALING:
                rowScaler = scalingTerm / static_cast<double>(numKmers);
                break;
            default:
                rowScaler = scalingTerm;
              break;
        }

        for (int j=0; j < numKmers; j++) {
            int repeatPosition = 0;
            xElement = inputPtr + i * xDim1 * xDim2 + j * xDim2;

            for (int k=0; k < numRepeats; k++) {
                for (int m=0; m < (convWidth * xDim2); m++)
                    copyBuffer[m] = xElement[m];
                for (int m=(convWidth * xDim2); m < paddedBufferSize; m++)
                    copyBuffer[m] = 0;

                SharedCPURandomFeatureOps::singleVectorSORF(copyBuffer,
                        rademPtr, repeatPosition, rademShape2,
                        paddedBufferSize);
                SharedCPURandomFeatureOps::singleVectorRBFPostGrad(copyBuffer,
                        chiPtr, outputPtr, gradientPtr, sigma,
                        paddedBufferSize, numFreqs, i, k, rowScaler);
                repeatPosition += paddedBufferSize;
            }
        }
    }
    delete[] copyBuffer;
    }

    return 0;
}
// Instantiate templates for use by wrapper.
template int convRBFGrad_<double>(
nb::ndarray<double, nb::shape<-1, -1, -1>, nb::device::cpu, nb::c_contig> inputArr,
nb::ndarray<double, nb::shape<-1, -1>, nb::device::cpu, nb::c_contig> outputArr,
nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cpu, nb::c_contig> radem,
nb::ndarray<double, nb::shape<-1>, nb::device::cpu, nb::c_contig> chiArr,
nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> seqlengths,
nb::ndarray<double, nb::shape<-1, -1,1>, nb::device::cpu, nb::c_contig> gradArr,
double sigma, int convWidth, int scalingType);

template int convRBFGrad_<float>(
nb::ndarray<float, nb::shape<-1, -1, -1>, nb::device::cpu, nb::c_contig> inputArr,
nb::ndarray<double, nb::shape<-1, -1>, nb::device::cpu, nb::c_contig> outputArr,
nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cpu, nb::c_contig> radem,
nb::ndarray<float, nb::shape<-1>, nb::device::cpu, nb::c_contig> chiArr,
nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> seqlengths,
nb::ndarray<double, nb::shape<-1, -1, 1>, nb::device::cpu, nb::c_contig> gradArr,
double sigma, int convWidth, int scalingType);



}  // namespace CPURBFConvolutionKernelCalculations
