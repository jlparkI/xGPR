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
#include "rbf_ops.h"
#include "../shared_fht_functions/hadamard_transforms.h"
#include "../shared_fht_functions/shared_rfgen_ops.h"

namespace nb = nanobind;


namespace CPURBFKernelCalculations {



/// @brief Calculates the random features only for the RBF kernel.
/// @param inputArr The input features that will be used to generate the RFs.
/// @param outputArr The array in which random features will be stored.
/// @param radem The array storing the diagonal Rademacher matrices.
/// @param chiArr The array storing the diagonal scaling matrix.
/// @param fitIntercept Whether to convert the first column to all 1s to fit
/// an intercept.
template <typename T>
int rbfFeatureGen_(nb::ndarray<T, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> inputArr,
        nb::ndarray<double, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> outputArr,
        nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cpu, nb::c_contig> radem,
        nb::ndarray<T, nb::shape<-1>, nb::device::cpu, nb::c_contig> chiArr,
        bool fitIntercept) {
    // Perform safety checks. Any exceptions thrown here are handed
    // off to Python by the Nanobind wrapper. We do not expect the user
    // to see these because the Python code will always ensure inputs
    // are correct -- these are a failsafe -- so we do not need to provide
    // detailed exception messages here.
    int xDim0 = inputArr.shape(0);
    int xDim1 = inputArr.shape(1);
    size_t rademShape2 = radem.shape(2);
    size_t numRffs = outputArr.shape(1);
    size_t numFreqs = chiArr.shape(0);
    double numFreqsFlt = numFreqs;

    T *inputPtr = static_cast<T*>(inputArr.data());
    double *outputPtr = static_cast<double*>(outputArr.data());
    T *chiPtr = static_cast<T*>(chiArr.data());
    int8_t *rademPtr = static_cast<int8_t*>(radem.data());

    if (inputArr.shape(0) == 0 || outputArr.shape(0) != inputArr.shape(0))
        throw std::runtime_error("no datapoints");
    if (numRffs < 2 || (numRffs & 1) != 0)
        throw std::runtime_error("last dim of output must be even number");
    if ( (2 * numFreqs) != numRffs || numFreqs > radem.shape(2) )
        throw std::runtime_error("incorrect number of rffs and or freqs.");

    double expectedNFreq = (xDim1 > 2) ? static_cast<double>(xDim1) : 2.0;
    double log2Freqs = std::log2(expectedNFreq);
    log2Freqs = std::ceil(log2Freqs);
    int paddedBufferSize = std::pow(2, log2Freqs);

    if (radem.shape(2) % paddedBufferSize != 0)
        throw std::runtime_error("incorrect number of rffs and or freqs.");

    T rbfNormConstant;

    if (fitIntercept)
        rbfNormConstant = std::sqrt(1.0 / (numFreqsFlt - 0.5));
    else
        rbfNormConstant = std::sqrt(1.0 / numFreqsFlt);


    #pragma omp parallel
    {
    int repeatPosition;
    int numRepeats = (numFreqs + paddedBufferSize - 1) / paddedBufferSize;
    // Notice that we don't have error handling here...very naughty. Out of
    // memory should be extremely rare since we are only allocating memory
    // for one row of the convolution. TODO: add error handling here.
    T *copyBuffer = new T[paddedBufferSize];

    #pragma omp for
    for (int i=0; i < xDim0; i++) {
        repeatPosition = 0;

        for (int k=0; k < numRepeats; k++) {
            int start_pos = i * xDim1;
            #pragma omp simd
            for (int m=0; m < xDim1; m++)
                copyBuffer[m] = inputPtr[start_pos + m];
            #pragma omp simd
            for (int m=xDim1; m < paddedBufferSize; m++)
                copyBuffer[m] = 0;

            SharedCPURandomFeatureOps::singleVectorSORF(copyBuffer,
                    rademPtr, repeatPosition, rademShape2, paddedBufferSize);
            SharedCPURandomFeatureOps::singleVectorRBFPostProcess(copyBuffer,
                    chiPtr, outputPtr, paddedBufferSize, numFreqs, i,
                    k, rbfNormConstant);
            repeatPosition += paddedBufferSize;
        }
    }
    delete[] copyBuffer;
    }

    return 0;
}
//Explicitly instantiate so wrapper can use.
template int rbfFeatureGen_<double>(nb::ndarray<double, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> inputArr,
nb::ndarray<double, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> outputArr,
nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cpu, nb::c_contig> radem,
nb::ndarray<double, nb::shape<-1>, nb::device::cpu, nb::c_contig> chiArr,
bool fitIntercept);

template int rbfFeatureGen_<float>(nb::ndarray<float, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> inputArr,
nb::ndarray<double, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> outputArr,
nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cpu, nb::c_contig> radem,
nb::ndarray<float, nb::shape<-1>, nb::device::cpu, nb::c_contig> chiArr,
bool fitIntercept);




/// @brief Calculates both the random features and the gradient for the
/// RBF kernel.
/// @param inputArr The input features that will be used to generate the RFs and
/// gradient.
/// @param outputArr The array in which random features will be stored.
/// @param precompWeights The precomputed weight matrix for converting from
/// input to random features.
/// @param radem The array storing the diagonal Rademacher matrices.
/// @param chiArr The array storing the diagonal scaling matrix.
/// @param sigma The lengthscale hyperparameter.
/// @param fitIntercept Whether to convert the first column to all 1s to fit
/// an intercept.
template <typename T>
int rbfGrad_(nb::ndarray<T, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> inputArr,
        nb::ndarray<double, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> outputArr,
        nb::ndarray<double, nb::shape<-1,-1,1>, nb::device::cpu, nb::c_contig> gradArr,
        nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cpu, nb::c_contig> radem,
        nb::ndarray<T, nb::shape<-1>, nb::device::cpu, nb::c_contig> chiArr,
        double sigma, bool fitIntercept) {
    // Perform safety checks. Any exceptions thrown here are handed
    // off to Python by the Nanobind wrapper. We do not expect the user
    // to see these because the Python code will always ensure inputs
    // are correct -- these are a failsafe -- so we do not need to provide
    // detailed exception messages here.
    int xDim0 = inputArr.shape(0);
    int xDim1 = inputArr.shape(1);
    size_t rademShape2 = radem.shape(2);
    size_t numRffs = outputArr.shape(1);
    size_t numFreqs = chiArr.shape(0);
    double numFreqsFlt = numFreqs;

    T *inputPtr = static_cast<T*>(inputArr.data());
    double *outputPtr = static_cast<double*>(outputArr.data());
    double *gradientPtr = static_cast<double*>(gradArr.data());
    T *chiPtr = static_cast<T*>(chiArr.data());
    int8_t *rademPtr = static_cast<int8_t*>(radem.data());

    if (inputArr.shape(0) == 0 || outputArr.shape(0) != inputArr.shape(0))
        throw std::runtime_error("no datapoints");
    if (numRffs < 2 || (numRffs & 1) != 0)
        throw std::runtime_error("last dim of output must be even number");
    if ( (2 * numFreqs) != numRffs || numFreqs > radem.shape(2) )
        throw std::runtime_error("incorrect number of rffs and or freqs.");
    if (gradArr.shape(0) != outputArr.shape(0) || gradArr.shape(1) != outputArr.shape(1))
        throw std::runtime_error("Wrong array sizes.");

    double expectedNFreq = (xDim1 > 2) ? static_cast<double>(xDim1) : 2.0;
    double log2Freqs = std::log2(expectedNFreq);
    log2Freqs = std::ceil(log2Freqs);
    int paddedBufferSize = std::pow(2, log2Freqs);

    if (radem.shape(2) % paddedBufferSize != 0)
        throw std::runtime_error("incorrect number of rffs and or freqs.");

    double rbfNormConstant;

    if (fitIntercept)
        rbfNormConstant = std::sqrt(1.0 / (numFreqsFlt - 0.5));
    else
        rbfNormConstant = std::sqrt(1.0 / numFreqsFlt);


    #pragma omp parallel
    {
    int repeatPosition;
    int numRepeats = (numFreqs + paddedBufferSize - 1) / paddedBufferSize;
    // Notice that we don't have error handling here...very naughty. Out of
    // memory should be extremely rare since we are only allocating memory
    // for one row of the convolution. TODO: add error handling here.
    T *copyBuffer = new T[paddedBufferSize];

    #pragma omp for
    for (int i=0; i < xDim0; i++) {
        repeatPosition = 0;

        for (int k=0; k < numRepeats; k++) {
            int start_pos = i * xDim1;
            #pragma omp simd
            for (int m=0; m < xDim1; m++)
                copyBuffer[m] = inputPtr[start_pos + m];
            #pragma omp simd
            for (int m=xDim1; m < paddedBufferSize; m++)
                copyBuffer[m] = 0;

            SharedCPURandomFeatureOps::singleVectorSORF(copyBuffer,
                    rademPtr, repeatPosition, rademShape2, paddedBufferSize);
            SharedCPURandomFeatureOps::singleVectorRBFPostGrad(copyBuffer,
                    chiPtr, outputPtr, gradientPtr, sigma, paddedBufferSize,
                    numFreqs, i, k, rbfNormConstant);
            repeatPosition += paddedBufferSize;
        }
    }
    delete[] copyBuffer;
    }

    return 0;
}
//Explicitly instantiate for external use.
template int rbfGrad_<double>(nb::ndarray<double, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> inputArr,
nb::ndarray<double, nb::shape<-1, -1>, nb::device::cpu, nb::c_contig> outputArr,
nb::ndarray<double, nb::shape<-1, -1, 1>, nb::device::cpu, nb::c_contig> gradArr,
nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cpu, nb::c_contig> radem,
nb::ndarray<double, nb::shape<-1>, nb::device::cpu, nb::c_contig> chiArr,
double sigma, bool fitIntercept);

template int rbfGrad_<float>(nb::ndarray<float, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> inputArr,
nb::ndarray<double, nb::shape<-1, -1>, nb::device::cpu, nb::c_contig> outputArr,
nb::ndarray<double, nb::shape<-1, -1, 1>, nb::device::cpu, nb::c_contig> gradArr,
nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cpu, nb::c_contig> radem,
nb::ndarray<float, nb::shape<-1>, nb::device::cpu, nb::c_contig> chiArr,
double sigma, bool fitIntercept);


}  // namespace CPURBFKernelCalculations
