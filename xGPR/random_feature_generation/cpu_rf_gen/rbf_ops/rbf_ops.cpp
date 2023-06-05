/*!
 * # double_rbf_ops.cpp
 *
 * This module performs all major steps involved in feature generation for
 * RBF-type kernels, which includes RBF, Matern, ARD and MiniARD (and by extension
 * the static layer kernels). Functions from array_operations are used to perform
 * the fast Hadamard transform and rademacher matrix multiplication pieces.
 * The "specialized" piece, multiplication by a diagonal matrix while performing
 * sine-cosine operations, is performed here.
 *
 * + rbfFeatureGen_
 * Performs the feature generation steps on an input array of doubles.
 *
 * + rbfGrad_
 * Performs the feature generation steps on an input array of doubles
 * AND generates the gradient info (stored in a separate array). For non-ARD
 * kernels only.
 *
 * + ardGrad_
 * Performs gradient and feature generation calculations for an RBF ARD kernel.
 * Slower than rbfFeatureGen, so use only if gradient is required.
 *
 * + ThreadRBFGen
 * Performs operations for a single thread of the feature generation operation.
 *
 * + ThreadRBFGrad
 * Performs operations for a single thread of the gradient / feature operation.
 * 
 * + ThreadARDGrad
 * Performs operations for a single thread of the ARD gradient-only calculation.
 *
 * + rbfFeatureGenLastStep_
 * Performs the final operations involved in the feature generation for doubles.
 *
 * + rbfGradLastStep_
 * Performs the final operations involved in feature / gradient calc for doubles.
 *
 * + ardGradCalcs
 * Performs the key operations involved in gradient-only calc for ARD.
 */
#include <Python.h>
#include <stdint.h>
#include <math.h>
#include <vector>
#include <thread>
#include "rbf_ops.h"
#include "../shared_fht_functions/basic_array_operations.h"


#define VALID_INPUTS 0
#define INVALID_INPUTS 1
#define EPS_TOLERANCE 0.0
#define MAX_THREADS 8


/*!
 * # rbfFeatureGen_
 *
 * Generates features for the input array and stores them in outputArray.
 *
 * ## Args:
 *
 * + `cArray` Pointer to the first element of the input array.
 * Must be a 3d array (N x D x C). C must be a power of 2.
 * + `radem` Pointer to first element of int8_t stack of diagonal.
 * arrays. Must be shape (3 x D x C).
 * + `chiArr` Pointer to first element of diagonal array to ensure
 * correct marginals. Must be of shape numFreqs.
 * + `outputArray` Pointer to first element of output array. Must
 * be a 2d array (N x 2 * numFreqs).
 * + `rbfNormConstant` A value by which all outputs are multipled.
 * Should be beta hparam * sqrt(1 / numFreqs). Is calculated by
 * caller.
 * + `dim0` shape[0] of input array
 * + `dim1` shape[1] of input array
 * + `dim2` shape[2] of input array
 * + `numFreqs` (numRFFs / 2) -- the number of frequencies to sample.
 * + `numThreads` The number of threads to use.
 */
template <typename T>
const char *rbfFeatureGen_(T cArray[], int8_t *radem,
                T chiArr[], double *outputArray,
                double rbfNormConstant,
                int dim0, int dim1, int dim2,
                int numFreqs, int numThreads){
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
        
        threads[i] = std::thread(&ThreadRBFGen<T>, cArray,
                radem, chiArr, outputArray, dim1, dim2,
                startPosition, endPosition, numFreqs,
                rbfNormConstant);
    }

    for (auto& th : threads)
        th.join();
    return "no_error";
}

/*!
 * # ThreadRBFGen
 *
 * Performs the RBF feature gen operation for one thread for a chunk of
 * the input array from startRow through endRow (each thread
 * works on its own group of rows).
 */
template <typename T>
void *ThreadRBFGen(T arrayStart[], int8_t *rademArray,
        T chiArr[], double *outputArray,
        int dim1, int dim2, int startPosition,
        int endPosition, int numFreqs, double rbfNormConstant){
    int rowSize = dim1 * dim2;

    multiplyByDiagonalRademacherMat<T>(arrayStart, rademArray,
                    dim1, dim2, startPosition, endPosition);
    transformRows3D<T>(arrayStart, startPosition, 
                    endPosition, dim1, dim2);

    multiplyByDiagonalRademacherMat<T>(arrayStart,
                    rademArray + rowSize, dim1, dim2, 
                    startPosition, endPosition);
    transformRows3D<T>(arrayStart, startPosition, 
                    endPosition, dim1, dim2);
    
    multiplyByDiagonalRademacherMat<T>(arrayStart,
                    rademArray + 2 * rowSize, dim1, dim2,
                    startPosition, endPosition);
    transformRows3D<T>(arrayStart, startPosition, 
                    endPosition, dim1, dim2);
    rbfFeatureGenLastStep_<T>(arrayStart, chiArr,
                    outputArray, rbfNormConstant,
                    startPosition, endPosition,
                    dim1, dim2, numFreqs);
    return NULL;
}




/*!
 * # rbfGrad_
 *
 * Generates features for the input array and stores them in outputArray.
 *
 * ## Args:
 *
 * + `cArray` Pointer to the first element of the input array.
 * Must be a 3d array (N x D x C). C must be a power of 2.
 * + `radem` Pointer to first element of int8_t stack of diagonal.
 * arrays. Must be shape (3 x D x C).
 * + `chiArr` Pointer to first element of diagonal array to ensure
 * correct marginals. Must be of shape numFreqs.
 * + `outputArray` Pointer to first element of output array. Must
 * be a 2d array (N x 2 * numFreqs).
 * + `gradientArray` Pointer to the first element of gradient array.
 * Must be a 2d array (N x 2 * numFreqs).
 * + `rbfNormConstant` A value by which all outputs are multipled.
 * Should be beta hparam * sqrt(1 / numFreqs). Is calculated by
 * caller.
 * + `sigma` The lengthscale hyperparameter.
 * + `dim0` shape[0] of input array
 * + `dim1` shape[1] of input array
 * + `dim2` shape[2] of input array
 * + `numFreqs` (numRFFs / 2) -- the number of frequencies to sample.
 * + `numThreads` The number of threads to use.
 */
template <typename T>
const char *rbfGrad_(T cArray[], int8_t *radem,
                T chiArr[], double *outputArray,
                double *gradientArray,
                double rbfNormConstant, T sigma,
                int dim0, int dim1, int dim2,
                int numFreqs, int numThreads){
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
 
        threads[i] = std::thread(&ThreadRBFGrad<T>, cArray,
                radem, chiArr, outputArray,
                gradientArray, dim1, dim2,
                startPosition, endPosition,
                numFreqs, rbfNormConstant, sigma);
    }

    for (auto& th : threads)
        th.join();
    return "no_error";
}


/*!
 * # ThreadRBFGrad
 *
 * Performs the RBF feature gen AND gradient operation for one thread
 * for a chunk of the input array from startRow through endRow (each thread
 * works on its own group of rows).
 */
template <typename T>
void *ThreadRBFGrad(T arrayStart[], int8_t* rademArray,
        T chiArr[], double *outputArray,
        double *gradientArray, int dim1, int dim2,
        int startPosition, int endPosition, int numFreqs,
        double rbfNormConstant, T sigma){
    int rowSize = dim1 * dim2;

    multiplyByDiagonalRademacherMat<T>(arrayStart,
                    rademArray,
                    dim1, dim2, 
                    startPosition, endPosition);
    transformRows3D<T>(arrayStart, startPosition, 
                    endPosition, dim1, dim2);

    multiplyByDiagonalRademacherMat<T>(arrayStart,
                    rademArray + rowSize,
                    dim1, dim2, 
                    startPosition, endPosition);
    transformRows3D<T>(arrayStart, startPosition, 
                    endPosition, dim1, dim2);
    
    multiplyByDiagonalRademacherMat<T>(arrayStart,
                    rademArray + 2 * rowSize,
                    dim1, dim2, 
                    startPosition, endPosition);
    transformRows3D<T>(arrayStart, startPosition, 
                    endPosition, dim1, dim2);
    rbfGradLastStep_<T>(arrayStart, chiArr,
                    outputArray, gradientArray,
                    rbfNormConstant, sigma,
                    startPosition, endPosition,
                    dim1, dim2, numFreqs);
    return NULL;
}


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







/*!
 * # ThreadARDGrad
 *
 * Performs ARD gradient-only calculations using pregenerated
 * features and weights.
 */
template <typename T>
void *ThreadARDGrad(T inputX[], double *randomFeats,
        T precompWeights[], int32_t *sigmaMap,
        double *sigmaVals, double *gradientArray,
        int startPosition, int endPosition,
        int dim1, int numLengthscales,
        int numFreqs, double rbfNormConstant){
    ardGradCalcs_<T>(inputX, randomFeats,
                    precompWeights, sigmaMap,
                    sigmaVals,
                    gradientArray, startPosition,
                    endPosition, dim1,
                    numLengthscales,
                    rbfNormConstant,
                    numFreqs);
    return NULL;
}



/*!
 * # rbfFeatureGenLastStep_
 *
 * Performs the last steps in RBF feature generation.
 *
 * ## Args:
 *
 * + `xArray` Pointer to the first element of the input array.
 * + `chiArray` Pointer to first element of diagonal array to ensure
 * correct marginals.
 * + `outputArray` Pointer to first element of output array.
 * + `rbfNormConstant` A value by which all outputs are multipled.
 * + `startRow` The starting row for this thread to work on.
 * + `endRow` The ending row for this thread to work on.
 * + `dim1` shape[1] of input array
 * + `dim2` shape[2] of input array
 * + `numFreqs` (numRFFs / 2) -- the number of frequencies to sample.
 */
template <typename T>
void rbfFeatureGenLastStep_(T xArray[], T chiArray[],
        double *outputArray, double normConstant,
        int startRow, int endRow, int dim1,
        int dim2, int numFreqs){
    int i, j;
    int elementsPerRow = dim1 * dim2;
    T *xElement;
    double *outputElement;
    T outputVal;

    for (i=startRow; i < endRow; i++){
        xElement = xArray + i * elementsPerRow;
        outputElement = outputArray + i * 2 * numFreqs;
        for (j=0; j < numFreqs; j++){
            outputVal = *xElement * chiArray[j];
            *outputElement = normConstant * cos(outputVal);
            outputElement++;
            *outputElement = normConstant * sin(outputVal);
            outputElement++;
            xElement++;
        }
    }
}



/*!
 * # rbfGradLastStep_
 *
 * Performs the last steps in RBF feature + gradient calcs.
 *
 * ## Args:
 *
 * + `xArray` Pointer to the first element of the input array.
 * + `chiArray` Pointer to first element of diagonal array to ensure
 * correct marginals.
 * + `outputArray` Pointer to first element of output array.
 * + `rbfNormConstant` A value by which all outputs are multipled.
 * + `startRow` The starting row for this thread to work on.
 * + `endRow` The ending row for this thread to work on.
 * + `dim1` shape[1] of input array
 * + `dim2` shape[2] of input array
 * + `numFreqs` (numRFFs / 2) -- the number of frequencies to sample.
 */
template <typename T>
void rbfGradLastStep_(T xArray[], T chiArray[],
        double *outputArray, double *gradientArray,
        double normConstant, T sigma,
        int startRow, int endRow, int dim1,
        int dim2, int numFreqs){
    int i, j;
    int elementsPerRow = dim1 * dim2;
    T *xElement;
    T outputVal;
    double *outputElement, *gradientElement;
    double cosVal, sinVal;

    for (i=startRow; i < endRow; i++){
        xElement = xArray + i * elementsPerRow;
        outputElement = outputArray + i * 2 * numFreqs;
        gradientElement = gradientArray + i * 2 * numFreqs;
        for (j=0; j < numFreqs; j++){
            outputVal = *xElement * chiArray[j];
            cosVal = cos(outputVal * sigma) * normConstant;
            sinVal = sin(outputVal * sigma) * normConstant;

            *outputElement = cosVal;
            outputElement++;
            *outputElement = sinVal;
            *gradientElement = -sinVal * outputVal;
            gradientElement++;
            *gradientElement = cosVal * outputVal;

            outputElement++;
            gradientElement++;
            xElement++;
        }
    }
}




/*!
 * # ardGradCalcs_
 *
 * Performs the key calculations for the miniARD gradient.
 *
 * ## Args:
 *
 * + `inputX` Pointer to the first element of the input array.
 * + `randomFeatures` Pointer to first element of random feature array.
 * + `precompWeights` Pointer to first element of precomputed weights.
 * + `sigmaMap` Pointer to first element of the array containing a
 * mapping from positions to lengthscales.
 * + `sigmaVals` Pointer to first element of shape (D) array containing the
 * per-feature lengthscales.
 * + `gradient` Pointer to the output array.
 * + `startRow` The starting row for this thread to work on.
 * + `endRow` The ending row for this thread to work on.
 * + `dim1` shape[1] of input array
 * + `numLengthscales` shape[2] of gradient
 * + `rbfNormConstant` A value by which all outputs are multipled.
 * Should be beta hparam * sqrt(1 / numFreqs). Is calculated by
 * caller.
 * + `numFreqs` (numRFFs / 2) -- the number of frequencies to sample.
 */
template <typename T>
void ardGradCalcs_(T inputX[], double *randomFeatures,
        T precompWeights[], int32_t *sigmaMap, double *sigmaVals,
        double *gradient, int startRow, int endRow, int dim1,
        int numLengthscales, double rbfNormConstant,
        int numFreqs){
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

            for (k=0; k < numLengthscales; k++){
                gradVal = gradientElement[k];
                gradientElement[k] = -gradVal * sinVal;
                gradientElement[k + numLengthscales] = gradVal * cosVal;
            }
            gradientElement += 2 * numLengthscales;
            randomFeature++;
        }
        xElement += dim1;
    }
}


//Explicitly instantiate so wrapper can use.
template const char *rbfFeatureGen_<double>(double cArray[], int8_t *radem,
                double chiArr[], double *outputArray,
                double rbfNormConstant,
                int dim0, int dim1, int dim2,
                int numFreqs, int numThreads);
template const char *rbfFeatureGen_<float>(float cArray[], int8_t *radem,
                float chiArr[], double *outputArray,
                double rbfNormConstant,
                int dim0, int dim1, int dim2,
                int numFreqs, int numThreads);

template const char *rbfGrad_<double>(double cArray[], int8_t *radem,
                double chiArr[], double *outputArray,
                double *gradientArray,
                double rbfNormConstant, double sigma,
                int dim0, int dim1, int dim2,
                int numFreqs, int numThreads);
template const char *rbfGrad_<float>(float cArray[], int8_t *radem,
                float chiArr[], double *outputArray,
                double *gradientArray,
                double rbfNormConstant, float sigma,
                int dim0, int dim1, int dim2,
                int numFreqs, int numThreads);

template const char *ardGrad_<double>(double inputX[], double *randomFeatures,
        double precompWeights[], int32_t *sigmaMap, double *sigmaVals,
        double *gradient, int dim0, int dim1, int numLengthscales,
        int numFreqs, double rbfNormConstant, int numThreads);
template const char *ardGrad_<float>(float inputX[], double *randomFeatures,
        float precompWeights[], int32_t *sigmaMap, double *sigmaVals,
        double *gradient, int dim0, int dim1, int numLengthscales,
        int numFreqs, double rbfNormConstant, int numThreads);
