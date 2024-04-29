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
#include "rbf_ops.h"
#include "../shared_fht_functions/hadamard_transforms.h"
#include "../shared_fht_functions/shared_rfgen_ops.h"




/*!
 * # ardGrad_
 *
 * Performs gradient-only calculations for the mini ARD kernel.
 *
 * ## Args:
 *
 * + `inputX` Pointer to the first element of the raw input data,
 * an (N x D) array.
 * + `randomFeatures` Pointer to first element of the array in which
 * random features will be stored, an (N x 2 * C) array.
 * + `precompWeights` Pointer to first element of the array containing
 * the precomputed weights, a (C x D) array.
 * + `sigmaMap` Pointer to first element of the array containing a mapping
 * from positions to lengthscales, a (D) array.
 * + `sigmaVals` Pointer to first element of shape (D) array containing the
 * per-feature lengthscales.
 * + `gradient` Pointer to first element of the array in which the gradient
 * will be stored, an (N x 2 * C) array.
 * + `dim0` shape[0] of input X
 * + `dim1` shape[1] of input X
 * + `numLengthscales` shape[2] of gradient
 * + `numFreqs` shape[0] of precompWeights
 * + `rbfNormConstant` A value by which all outputs are multipled.
 * Should be beta hparam * sqrt(1 / numFreqs). Is calculated by
 * caller.
 * + `numThreads` The number of threads to use.
 */
template <typename T>
const char *ardGrad_(T inputX[], double *randomFeatures,
        T precompWeights[], int32_t *sigmaMap, double *sigmaVals,
        double *gradient, int dim0, int dim1, int numLengthscales,
        int numFreqs, double rbfNormConstant, int numThreads){
    if (numThreads > dim0)
        numThreads = dim0;

    std::vector<std::thread> threads(numThreads);
    int startPosition, endPosition;
    int chunkSize = (dim0 + numThreads - 1) / numThreads;

    for (int i=0; i < numThreads; i++){
        startPosition = i * chunkSize;
        endPosition = (i + 1) * chunkSize;
        if (endPosition > dim0)
            endPosition = dim0;
 
        threads[i] = std::thread(&ThreadARDGrad<T>, inputX, randomFeatures,
                precompWeights, sigmaMap, sigmaVals, gradient,
                startPosition, endPosition, dim1,
                numLengthscales, numFreqs, rbfNormConstant);
    }

    for (auto& th : threads)
        th.join();
    return "no_error";
}
template const char *ardGrad_<double>(double inputX[], double *randomFeatures,
        double precompWeights[], int32_t *sigmaMap, double *sigmaVals,
        double *gradient, int dim0, int dim1, int numLengthscales,
        int numFreqs, double rbfNormConstant, int numThreads);
template const char *ardGrad_<float>(float inputX[], double *randomFeatures,
        float precompWeights[], int32_t *sigmaMap, double *sigmaVals,
        double *gradient, int dim0, int dim1, int numLengthscales,
        int numFreqs, double rbfNormConstant, int numThreads);







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
