/* Copyright (C) 2025 Jonathan Parkinson
*/
// C++ headers
#include <math.h>
#include <vector>
#include <algorithm>

// Library headers

// Project headers
#include "rbf_convolution.h"
#include "../shared_fht_functions/hadamard_transforms.h"
#include "../shared_fht_functions/shared_rfgen_ops.h"




namespace CPURBFConvolutionKernelCalculations {



template <typename T>
int convRBFFeatureGen_(nb::ndarray<T, nb::shape<-1, -1, -1>, nb::device::cpu, nb::c_contig> input_arr,
nb::ndarray<double, nb::shape<-1, -1>, nb::device::cpu, nb::c_contig> output_arr,
nb::ndarray<int8_t, nb::shape<3,1, -1>, nb::device::cpu, nb::c_contig> radem,
nb::ndarray<T, nb::shape<-1>, nb::device::cpu, nb::c_contig> chi_arr,
nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> seqlengths,
int conv_width, int scaling_type) {
    // Perform safety checks. Any exceptions thrown here are handed off
    // to Python by the Nanobind wrapper. We do not expect the user to
    // see these because the Python code will always ensure inputs are
    // correct -- these are a failsafe -- so we do not need to provide
    // detailed exception messages here.
    int xDim0 = input_arr.shape(0);
    int xDim1 = input_arr.shape(1);
    int xDim2 = input_arr.shape(2);
    int rademShape2 = radem.shape(2);
    size_t numRffs = output_arr.shape(1);
    size_t numFreqs = chi_arr.shape(0);
    double scalingTerm = std::sqrt(1.0 / static_cast<double>(numFreqs));

    T *inputPtr = static_cast<T*>(input_arr.data());
    double *outputPtr = static_cast<double*>(output_arr.data());
    T *chiPtr = static_cast<T*>(chi_arr.data());
    int8_t *rademPtr = static_cast<int8_t*>(radem.data());
    int32_t *seqlengthsPtr = static_cast<int32_t*>(seqlengths.data());

    if (input_arr.shape(0) == 0 || output_arr.shape(0) != input_arr.shape(0))
        throw std::runtime_error("no datapoints");
    if (numRffs < 2 || (numRffs & 1) != 0)
        throw std::runtime_error("last dim of output must be even number");
    if ( (2 * numFreqs) != numRffs || numFreqs > radem.shape(2) )
        throw std::runtime_error("incorrect number of rffs and or freqs.");

    if (seqlengths.shape(0) != input_arr.shape(0))
        throw std::runtime_error("wrong array sizes");
    if (static_cast<int>(input_arr.shape(1)) < conv_width || conv_width <= 0)
        throw std::runtime_error("invalid conv_width");

    double expectedNFreq = static_cast<double>(conv_width * input_arr.shape(2));
    expectedNFreq = std::max(expectedNFreq, 2.);
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

    if (maxSeqLength > static_cast<int32_t>(input_arr.shape(1)) ||
            minSeqLength < conv_width) {
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
        numKmers = seqlength - conv_width + 1;
        switch (scaling_type) {
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
                for (int m=0; m < (conv_width * xDim2); m++)
                    copyBuffer[m] = xElement[m];
                #pragma omp simd
                for (int m=(conv_width * xDim2); m < paddedBufferSize; m++)
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
template int convRBFFeatureGen_<double>(nb::ndarray<double, nb::shape<-1, -1, -1>, nb::device::cpu, nb::c_contig> input_arr,
nb::ndarray<double, nb::shape<-1, -1>, nb::device::cpu, nb::c_contig> output_arr,
nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cpu, nb::c_contig> radem,
nb::ndarray<double, nb::shape<-1>, nb::device::cpu, nb::c_contig> chi_arr,
nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> seqlengths,
int conv_width, int scaling_type);

template int convRBFFeatureGen_<float>(nb::ndarray<float, nb::shape<-1, -1, -1>, nb::device::cpu, nb::c_contig> input_arr,
nb::ndarray<double, nb::shape<-1, -1>, nb::device::cpu, nb::c_contig> output_arr,
nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cpu, nb::c_contig> radem,
nb::ndarray<float, nb::shape<-1>, nb::device::cpu, nb::c_contig> chi_arr,
nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> seqlengths,
int conv_width, int scaling_type);





template <typename T>
int convRBFGrad_(nb::ndarray<T, nb::shape<-1, -1, -1>, nb::device::cpu, nb::c_contig> input_arr,
nb::ndarray<double, nb::shape<-1, -1>, nb::device::cpu, nb::c_contig> output_arr,
nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cpu, nb::c_contig> radem,
nb::ndarray<T, nb::shape<-1>, nb::device::cpu, nb::c_contig> chi_arr,
nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> seqlengths,
nb::ndarray<double, nb::shape<-1, -1,1>, nb::device::cpu, nb::c_contig> grad_arr,
double sigma, int conv_width, int scaling_type) {
    // Perform safety checks. Any exceptions thrown here are handed off
    // to Python by the Nanobind wrapper. We do not expect the user to
    // see these because the Python code will always ensure inputs are
    // correct -- these are a failsafe -- so we do not need to provide
    // detailed exception messages here.
    int xDim0 = input_arr.shape(0);
    int xDim1 = input_arr.shape(1);
    int xDim2 = input_arr.shape(2);
    int rademShape2 = radem.shape(2);
    size_t numRffs = output_arr.shape(1);
    size_t numFreqs = chi_arr.shape(0);
    double scalingTerm = std::sqrt(1.0 / static_cast<double>(numFreqs));

    T *inputPtr = static_cast<T*>(input_arr.data());
    double *outputPtr = static_cast<double*>(output_arr.data());
    T *chiPtr = static_cast<T*>(chi_arr.data());
    int8_t *rademPtr = static_cast<int8_t*>(radem.data());
    int32_t *seqlengthsPtr = static_cast<int32_t*>(seqlengths.data());
    double *gradientPtr = static_cast<double*>(grad_arr.data());

    if (input_arr.shape(0) == 0 || output_arr.shape(0) != input_arr.shape(0))
        throw std::runtime_error("no datapoints");
    if (numRffs < 2 || (numRffs & 1) != 0)
        throw std::runtime_error("last dim of output must be even number");
    if ( (2 * numFreqs) != numRffs || numFreqs > radem.shape(2) )
        throw std::runtime_error("incorrect number of rffs and or freqs.");

    if (seqlengths.shape(0) != input_arr.shape(0))
        throw std::runtime_error("wrong array sizes");
    if (static_cast<int>(input_arr.shape(1)) < conv_width || conv_width <= 0)
        throw std::runtime_error("invalid conv_width");

    if (grad_arr.shape(0) != output_arr.shape(0) ||
            grad_arr.shape(1) != output_arr.shape(1))
        throw std::runtime_error("wrong array sizes");

    double expectedNFreq = static_cast<double>(conv_width * input_arr.shape(2));
    expectedNFreq = std::max(expectedNFreq, 2.);
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

    if (maxSeqLength > static_cast<int32_t>(input_arr.shape(1))
            || minSeqLength < conv_width) {
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
        numKmers = seqlength - conv_width + 1;
        switch (scaling_type) {
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
                for (int m=0; m < (conv_width * xDim2); m++)
                    copyBuffer[m] = xElement[m];
                for (int m=(conv_width * xDim2); m < paddedBufferSize; m++)
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
nb::ndarray<double, nb::shape<-1, -1, -1>, nb::device::cpu, nb::c_contig> input_arr,
nb::ndarray<double, nb::shape<-1, -1>, nb::device::cpu, nb::c_contig> output_arr,
nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cpu, nb::c_contig> radem,
nb::ndarray<double, nb::shape<-1>, nb::device::cpu, nb::c_contig> chi_arr,
nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> seqlengths,
nb::ndarray<double, nb::shape<-1, -1,1>, nb::device::cpu, nb::c_contig> grad_arr,
double sigma, int conv_width, int scaling_type);

template int convRBFGrad_<float>(
nb::ndarray<float, nb::shape<-1, -1, -1>, nb::device::cpu, nb::c_contig> input_arr,
nb::ndarray<double, nb::shape<-1, -1>, nb::device::cpu, nb::c_contig> output_arr,
nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cpu, nb::c_contig> radem,
nb::ndarray<float, nb::shape<-1>, nb::device::cpu, nb::c_contig> chi_arr,
nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> seqlengths,
nb::ndarray<double, nb::shape<-1, -1, 1>, nb::device::cpu, nb::c_contig> grad_arr,
double sigma, int conv_width, int scaling_type);



}  // namespace CPURBFConvolutionKernelCalculations
