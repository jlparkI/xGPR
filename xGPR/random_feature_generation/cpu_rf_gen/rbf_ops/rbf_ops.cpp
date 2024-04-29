/*!
 * # double_rbf_ops.cpp
 *
 * This module performs all major steps involved in feature generation for
 * RBF-type kernels, which includes RBF, Matern, ARD and MiniARD (and by extension
 * the static layer kernels).
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
                int dim0, int dim1, int rademShape2,
                int numFreqs, int numThreads,
                int paddedBufferSize){
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
        
        threads[i] = std::thread(&allInOneRBFGen<T>, cArray,
                radem, chiArr, outputArray, dim1, numFreqs,
                rademShape2, startPosition, endPosition,
                paddedBufferSize, rbfNormConstant);
    }

    for (auto& th : threads)
        th.join();
    return "no_error";
}
//Explicitly instantiate so wrapper can use.
template const char *rbfFeatureGen_<double>(double cArray[], int8_t *radem,
                double chiArr[], double *outputArray,
                double rbfNormConstant,
                int dim0, int dim1, int rademShape2,
                int numFreqs, int numThreads,
                int paddedBufferSize);
template const char *rbfFeatureGen_<float>(float cArray[], int8_t *radem,
                float chiArr[], double *outputArray,
                double rbfNormConstant,
                int dim0, int dim1, int rademShape2,
                int numFreqs, int numThreads,
                int paddedBufferSize);


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
 * + `rademShape2` shape[2] of radem
 * + `numFreqs` (numRFFs / 2) -- the number of frequencies to sample.
 * + `numThreads` The number of threads to use.
 */
template <typename T>
const char *rbfGrad_(T cArray[], int8_t *radem,
                T chiArr[], double *outputArray,
                double *gradientArray,
                double rbfNormConstant, T sigma,
                int dim0, int dim1, int rademShape2,
                int numFreqs, int numThreads,
                int paddedBufferSize){
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
 
        threads[i] = std::thread(&allInOneRBFGrad<T>, cArray,
                radem, chiArr, outputArray,
                gradientArray, dim1, numFreqs,
                rademShape2, startPosition,
                endPosition, paddedBufferSize,
                rbfNormConstant, sigma);
    }

    for (auto& th : threads)
        th.join();
    return "no_error";
}
//Explicitly instantiate for external use.
template const char *rbfGrad_<double>(double cArray[], int8_t *radem,
                double chiArr[], double *outputArray,
                double *gradientArray,
                double rbfNormConstant, double sigma,
                int dim0, int dim1, int rademShape2,
                int numFreqs, int numThreads,
                int paddedBufferSize);
template const char *rbfGrad_<float>(float cArray[], int8_t *radem,
                float chiArr[], double *outputArray,
                double *gradientArray,
                double rbfNormConstant, float sigma,
                int dim0, int dim1, int rademShape2,
                int numFreqs, int numThreads,
                int paddedBufferSize);




/*!
 * # allInOneRBFGen
 *
 * Performs the RBF-based kernel feature generation
 * process for the input, for one thread.
 */
template <typename T>
void *allInOneRBFGen(T xdata[], int8_t *rademArray, T chiArr[],
        double *outputArray, int dim1, int numFreqs, int rademShape2,
        int startRow, int endRow, int paddedBufferSize,
        double scalingTerm){

    int numRepeats = (numFreqs + paddedBufferSize - 1) / paddedBufferSize;
    //Notice that we don't have error handling here...very naughty. Out of
    //memory should be extremely rare since we are only allocating memory
    //for one row of the convolution. TODO: add error handling here.
    T *copyBuffer = new T[paddedBufferSize];
    T *xElement;

    for (int i=startRow; i < endRow; i++){

        int repeatPosition = 0;
        xElement = xdata + i * dim1;

        for (int k=0; k < numRepeats; k++){
            for (int m=0; m < dim1; m++)
                copyBuffer[m] = xElement[m];
            for (int m=dim1; m < paddedBufferSize; m++)
                copyBuffer[m] = 0;

            singleVectorSORF(copyBuffer, rademArray, repeatPosition,
                        rademShape2, paddedBufferSize);
            singleVectorRBFPostProcess(copyBuffer, chiArr, outputArray,
                        paddedBufferSize, numFreqs, i, k, scalingTerm);
            repeatPosition += paddedBufferSize;
        }
    }
    delete[] copyBuffer;

    return NULL;
}


/*!
 * # allInOneRBFGrad
 *
 * Performs the RBF-based kernel feature generation
 * process for the input, for one thread, and calculates the
 * gradient, which is stored in a separate array.
 */
template <typename T>
void *allInOneRBFGrad(T xdata[], int8_t *rademArray, T chiArr[],
        double *outputArray, double *gradientArray,
        int dim1, int numFreqs, int rademShape2, int startRow,
        int endRow, int paddedBufferSize,
        double scalingTerm, T sigma){

    int numRepeats = (numFreqs + paddedBufferSize - 1) / paddedBufferSize;
    //Notice that we don't have error handling here...very naughty. Out of
    //memory should be extremely rare since we are only allocating memory
    //for one row of the convolution. TODO: add error handling here.
    T *copyBuffer = new T[paddedBufferSize];
    T *xElement;

    for (int i=startRow; i < endRow; i++){
        int repeatPosition = 0;
        xElement = xdata + i * dim1;

        for (int k=0; k < numRepeats; k++){
            for (int m=0; m < dim1; m++)
                copyBuffer[m] = xElement[m];
            for (int m=dim1; m < paddedBufferSize; m++)
                copyBuffer[m] = 0;

            singleVectorSORF(copyBuffer, rademArray, repeatPosition,
                        rademShape2, paddedBufferSize);
            singleVectorRBFPostGrad(copyBuffer, chiArr, outputArray,
                        gradientArray, sigma, paddedBufferSize, numFreqs,
                        i, k, scalingTerm);
            repeatPosition += paddedBufferSize;
        }
    }
    delete[] copyBuffer;

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
