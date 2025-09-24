/* Copyright (C) 2025 Jonathan Parkinson
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
// C++ headers
#include <stdint.h>
#include <math.h>
#include <vector>

// Library headers

// Project headers
#include "ard_ops.h"
#include "../shared_fht_functions/hadamard_transforms.h"
#include "../shared_fht_functions/shared_rfgen_ops.h"

namespace nb = nanobind;


namespace CPUARDKernelCalculations {


/// @brief Calculates both the random features and the gradient for the MiniARD
/// kernel. TODO: This function needs further optimization.
/// @param inputArr The input features that will be used to generate the RFs and
/// gradient.
/// @param outputArr The array in which random features will be stored.
/// @param precompWeights The precomputed weight matrix for converting from
/// input to random features.
/// @param sigmaMap A map of which sigma value is applicable for
/// which input values.
/// @param sigmaVals The actual feature or region specific sigma hyperparameters
/// (lengthscales).
/// @param gradArr The array in which the calculated gradient will be stored.
/// @param fitIntercept Whether to convert the first column to all 1s to fit an
/// intercept.
template <typename T>
int ardGrad_(nb::ndarray<T, nb::shape<-1, -1>, nb::device::cpu, nb::c_contig> inputArr,
        nb::ndarray<double, nb::shape<-1, -1>, nb::device::cpu, nb::c_contig> outputArr,
        nb::ndarray<T, nb::shape<-1, -1>, nb::device::cpu, nb::c_contig> precompWeights,
        nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> sigmaMap,
        nb::ndarray<double, nb::shape<-1>, nb::device::cpu, nb::c_contig> sigmaVals,
        nb::ndarray<double, nb::shape<-1, -1, -1>, nb::device::cpu, nb::c_contig> gradArr,
        bool fitIntercept) {
    // Perform safety checks. Any exceptions thrown here are
    // handed off to Python by the Nanobind wrapper. We do not
    // expect the user to see these because the Python code will
    // always ensure inputs are correct -- these are a failsafe --
    // so we do not need to provide detailed exception messages here.
    int zDim0 = inputArr.shape(0);
    int zDim1 = inputArr.shape(1);

    T *inputPtr = static_cast<T*>(inputArr.data());
    T *precompWeightsPtr = static_cast<T*>(precompWeights.data());
    double *outputPtr = static_cast<double*>(outputArr.data());
    double *gradientPtr = static_cast<double*>(gradArr.data());
    int32_t *sigmaMapPtr = static_cast<int32_t*>(sigmaMap.data());
    double *sigmaValsPtr = static_cast<double*>(sigmaVals.data());

    size_t numFreqs = precompWeights.shape(0);
    double numFreqsFlt = numFreqs;
    size_t numLengthscales = gradArr.shape(2);

    if (inputArr.shape(0) == 0 || outputArr.shape(0) != inputArr.shape(0))
        throw std::runtime_error("no datapoints");

    if (gradArr.shape(0) != outputArr.shape(0) ||
            gradArr.shape(1) != outputArr.shape(1))
        throw std::runtime_error("Wrong array sizes.");

    if (precompWeights.shape(1) != inputArr.shape(1))
        throw std::runtime_error("Wrong array sizes.");

    if (outputArr.shape(1) != 2 * precompWeights.shape(0) ||
            sigmaMap.shape(0) != precompWeights.shape(1))
        throw std::runtime_error("Wrong array sizes.");

    if (sigmaVals.shape(0) != sigmaMap.shape(0))
        throw std::runtime_error("Wrong array sizes.");


    T rbfNormConstant;

    if (fitIntercept)
        rbfNormConstant = std::sqrt(1.0 / (numFreqsFlt - 0.5));
    else
        rbfNormConstant = std::sqrt(1.0 / numFreqsFlt);

    int gradIncrement = numFreqs * numLengthscales;

    for (int i=0; i < zDim0; i++) {
        T *precompWeight = precompWeightsPtr;
        double *gradientElement = gradientPtr + i * 2 * gradIncrement;
        double *randomFeature = outputPtr + i * numFreqs * 2;

        for (int j=0; j < numFreqs; j++) {
            double rfSum = 0;

            for (int k=0; k < zDim1; k++) {
                double dotProd = inputPtr[k] * *precompWeight;
                gradientElement[sigmaMapPtr[k]] += dotProd;
                rfSum += sigmaValsPtr[k] * dotProd;
                precompWeight++;
            }

            double cosVal = rbfNormConstant * cos(rfSum);
            double sinVal = rbfNormConstant * sin(rfSum);
            *randomFeature = cosVal;
            randomFeature++;
            *randomFeature = sinVal;
            randomFeature++;

            for (int k=0; k < numLengthscales; k++) {
                double gradVal = gradientElement[k];
                gradientElement[k] = -gradVal * sinVal;
                gradientElement[k + numLengthscales] = gradVal * cosVal;
            }
            gradientElement += 2 * numLengthscales;
        }
        inputPtr += zDim1;
    }
    return 0;
}
template int ardGrad_<double>(
nb::ndarray<double, nb::shape<-1, -1>, nb::device::cpu, nb::c_contig> inputArr,
nb::ndarray<double, nb::shape<-1, -1>, nb::device::cpu, nb::c_contig> outputArr,
nb::ndarray<double, nb::shape<-1, -1>, nb::device::cpu, nb::c_contig> precompWeights,
nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> sigmaMap,
nb::ndarray<double, nb::shape<-1>, nb::device::cpu, nb::c_contig> sigmaVals,
nb::ndarray<double, nb::shape<-1, -1, -1>, nb::device::cpu, nb::c_contig> gradArr,
bool fitIntercept);

template int ardGrad_<float>(
nb::ndarray<float, nb::shape<-1, -1>, nb::device::cpu, nb::c_contig> inputArr,
nb::ndarray<double, nb::shape<-1, -1>, nb::device::cpu, nb::c_contig> outputArr,
nb::ndarray<float, nb::shape<-1, -1>, nb::device::cpu, nb::c_contig> precompWeights,
nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> sigmaMap,
nb::ndarray<double, nb::shape<-1>, nb::device::cpu, nb::c_contig> sigmaVals,
nb::ndarray<double, nb::shape<-1, -1, -1>, nb::device::cpu, nb::c_contig> gradArr,
bool fitIntercept);



}  // namespace CPUARDKernelCalculations
