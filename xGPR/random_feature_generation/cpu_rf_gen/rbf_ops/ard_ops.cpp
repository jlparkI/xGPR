/*!
 * # ard_ops.cpp
 * 
 * This module performs major steps involved in calculating gradients for
 * ARD kernels (a somewhat specialized task). TODO: Further optimization
 * needed for this module.
 */
#include <Python.h>
#include <stdint.h>
#include <math.h>
#include <vector>
#include <thread>
#include "ard_ops.h"
#include "../shared_fht_functions/hadamard_transforms.h"
#include "../shared_fht_functions/shared_rfgen_ops.h"

namespace nb = nanobind;



/*!
 * # ardGrad_
 *
 * Performs gradient-only calculations for the mini ARD kernel.
 *
 * ## Args:
 *
 * + `inputArr` A numpy array of shape (N x C).
 * + `outputArr` A numpy array of shape (N x R),
 * where R is the number of RFFs and is 2x numFreqs;
 * + `precompWeights` Numpy array containing the precomputed
 * weights, a (C x D) array.
 * + `sigmaMap` Numpy array containing a mapping from positions
 * to lengthscales, a (D) array.
 * + `sigmaVals` A shape (D) numpy array containing the per-feature
 * lengthscales.
 * + `gradArr` The (N x R x L) numpy array for the gradient.
 * + `numThreads` The number of threads to use.
 * + `fitIntercept` Whether an intercept will be fitted.
 */
template <typename T>
int ardGrad_(nb::ndarray<T, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> inputArr,
        nb::ndarray<double, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> outputArr,
        nb::ndarray<T, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> precompWeights,
        nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> sigmaMap,
        nb::ndarray<double, nb::shape<-1>, nb::device::cpu, nb::c_contig> sigmaVals,
        nb::ndarray<double, nb::shape<-1,-1,-1>, nb::device::cpu, nb::c_contig> gradArr,
        int numThreads, bool fitIntercept){
    // Perform safety checks. Any exceptions thrown here are handed off to Python
    // by the Nanobind wrapper. We do not expect the user to see these because
    // the Python code will always ensure inputs are correct -- these are a failsafe
    // -- so we do not need to provide detailed exception messages here.
    int zDim0 = inputArr.shape(0);

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
    if (gradArr.shape(0) != outputArr.shape(0) || gradArr.shape(1) != outputArr.shape(1))
        throw std::runtime_error("Wrong array sizes.");
    if (precompWeights.shape(1) != inputArr.shape(1))
        throw std::runtime_error("Wrong array sizes.");
    if (outputArr.shape(1) != 2 * precompWeights.shape(0) || sigmaMap.shape(0) != precompWeights.shape(1))
        throw std::runtime_error("Wrong array sizes.");
    if (sigmaVals.shape(0) != sigmaMap.shape(0))
        throw std::runtime_error("Wrong array sizes.");


    T rbfNormConstant;

    if (fitIntercept)
        rbfNormConstant = std::sqrt(1.0 / (numFreqsFlt - 0.5));
    else
        rbfNormConstant = std::sqrt(1.0 / numFreqsFlt);

    if (numThreads > zDim0)
        numThreads = zDim0;

    std::vector<std::thread> threads(numThreads);
    int startPosition, endPosition;
    int chunkSize = (zDim0 + numThreads - 1) / numThreads;

    for (int i=0; i < numThreads; i++){
        startPosition = i * chunkSize;
        endPosition = (i + 1) * chunkSize;
        if (endPosition > zDim0)
            endPosition = zDim0;
 
        threads[i] = std::thread(&ThreadARDGrad<T>, inputPtr, outputPtr,
                precompWeightsPtr, sigmaMapPtr, sigmaValsPtr, gradientPtr,
                startPosition, endPosition, inputArr.shape(1),
                numLengthscales, numFreqs, rbfNormConstant);
    }

    for (auto& th : threads)
        th.join();
    return 0;
}
template int ardGrad_<double>(nb::ndarray<double, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> inputArr,
        nb::ndarray<double, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> outputArr,
        nb::ndarray<double, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> precompWeights,
        nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> sigmaMap,
        nb::ndarray<double, nb::shape<-1>, nb::device::cpu, nb::c_contig> sigmaVals,
        nb::ndarray<double, nb::shape<-1,-1,-1>, nb::device::cpu, nb::c_contig> gradArr,
        int numThreads, bool fitIntercept);
template int ardGrad_<float>(nb::ndarray<float, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> inputArr,
        nb::ndarray<double, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> outputArr,
        nb::ndarray<float, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> precompWeights,
        nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> sigmaMap,
        nb::ndarray<double, nb::shape<-1>, nb::device::cpu, nb::c_contig> sigmaVals,
        nb::ndarray<double, nb::shape<-1,-1,-1>, nb::device::cpu, nb::c_contig> gradArr,
        int numThreads, bool fitIntercept);







/*!
 * # ThreadARDGrad
 *
 * Performs ARD gradient-only calculations using pregenerated
 * features and weights.
 */
template <typename T>
void *ThreadARDGrad(T inputX[], double *randomFeatures,
        T precompWeights[], int32_t *sigmaMap,
        double *sigmaVals, double *gradient,
        int startRow, int endRow,
        int dim1, int numLengthscales,
        int numFreqs, double rbfNormConstant){
    int i, j, k;
    int gradIncrement = numFreqs * numLengthscales;
    T *xElement, *precompWeight;
    double dotProd;
    double *gradientElement, *randomFeature;
    double gradVal, sinVal, cosVal, rfSum;

    xElement = inputX + startRow * dim1;

    for (i=startRow; i < endRow; i++){
        precompWeight = precompWeights;
        gradientElement = gradient + i * 2 * gradIncrement;
        randomFeature = randomFeatures + i * numFreqs * 2;

        for (j=0; j < numFreqs; j++){
            rfSum = 0;

            for (k=0; k < dim1; k++){
                dotProd = xElement[k] * *precompWeight;
                gradientElement[sigmaMap[k]] += dotProd;
                rfSum += sigmaVals[k] * dotProd;
                precompWeight++;
            }

            cosVal = rbfNormConstant * cos(rfSum);
            sinVal = rbfNormConstant * sin(rfSum);
            *randomFeature = cosVal;
            randomFeature++;
            *randomFeature = sinVal;
            randomFeature++;

            for (k=0; k < numLengthscales; k++){
                gradVal = gradientElement[k];
                gradientElement[k] = -gradVal * sinVal;
                gradientElement[k + numLengthscales] = gradVal * cosVal;
            }
            gradientElement += 2 * numLengthscales;
        }
        xElement += dim1;
    }
    return NULL;
}
