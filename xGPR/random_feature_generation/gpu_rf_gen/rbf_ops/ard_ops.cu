/*
* Contains specialized functions for generating random features for
* ARD RBF kernels (non-convolution).
*/

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#include "../shared_constants.h"
#include "../sharedmem.h"
#include "ard_ops.h"



//Performs the first piece of the gradient calculation for ARD kernels
//only -- multiplying the input data by the precomputed weight matrix
//and summing over rows that correspond to specific lengthscales.
template <typename T>
__global__ void ardGradSetup(double *gradientArray,
        T precomputedWeights[], T inputX[], int32_t *sigmaMap,
        double *sigmaVals, double *randomFeatures,
        int dim1, int numSetupElements, int numFreqs,
        int numLengthscales){

    int i, sigmaLoc;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int precompWRow = (tid % numFreqs);
    int gradRow = tid / numFreqs;

    T outVal;

    if (tid < numSetupElements){
        T *precompWElement = precomputedWeights + precompWRow * dim1;
        T *inputXElement = inputX + gradRow * dim1;
        double *gradientElement = gradientArray + 2 * (gradRow * numFreqs + precompWRow) * numLengthscales;
        double *randomFeature = randomFeatures + 2 * (gradRow * numFreqs + precompWRow);
        double rfVal = 0;

        for (i=0; i < dim1; i++){
            sigmaLoc = sigmaMap[i];
            outVal = precompWElement[i] * inputXElement[i];
            gradientElement[sigmaLoc] += outVal;
            rfVal += sigmaVals[i] * outVal;
        }
        *randomFeature = rfVal;
    }
}





//Multiplies the gradient array by the appropriate elements of the random
//feature array when calculating the gradient for ARD kernels only.
__global__ void ardGradRFMultiply(double *gradientArray, double *randomFeats,
        int numRFElements, int numFreqs, int numLengthscales,
        double rbfNormConstant){
    int i;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int rowNum = tid / numFreqs, colNum = tid % numFreqs;
    int gradPosition = 2 * (rowNum * numFreqs + colNum) * numLengthscales;
    int rfPosition = 2 * (rowNum * numFreqs + colNum);
    double rfVal, cosVal, sinVal;
    

    if (tid < numRFElements){
        rfVal = randomFeats[rfPosition];
        cosVal = cos(rfVal) * rbfNormConstant;
        sinVal = sin(rfVal) * rbfNormConstant;
        randomFeats[rfPosition] = cosVal;
        randomFeats[rfPosition + 1] = sinVal;

        for (i=0; i < numLengthscales; i++){
            rfVal = gradientArray[gradPosition + i];
            gradientArray[gradPosition + i] = -rfVal * sinVal;
            gradientArray[gradPosition + i + numLengthscales] = rfVal * cosVal;
        }
    }
}


//This function generates the gradient and random features
//for ARD kernels only, using precomputed weights that take
//the place of the H-transforms
//we would otherwise need to perform.
template <typename T>
const char *ardCudaGrad(T inputX[], double *randomFeats,
                T precompWeights[], int32_t *sigmaMap,
                double *sigmaVals, double *gradient, int dim0,
                int dim1, int numLengthscales, int numFreqs,
                double rbfNormConstant){

    int numRFElements = dim0 * numFreqs;
    int numSetupElements = dim0 * numFreqs;
    int blocksPerGrid;


    blocksPerGrid = (numSetupElements + DEFAULT_THREADS_PER_BLOCK - 1) / DEFAULT_THREADS_PER_BLOCK;
    ardGradSetup<T><<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(gradient, precompWeights, inputX,
            sigmaMap, sigmaVals, randomFeats, dim1, numSetupElements,
            numFreqs, numLengthscales);

    blocksPerGrid = (numRFElements + DEFAULT_THREADS_PER_BLOCK - 1) / DEFAULT_THREADS_PER_BLOCK;
    ardGradRFMultiply<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(gradient, randomFeats,
                numRFElements, numFreqs, numLengthscales, rbfNormConstant);

    return "no_error";
}
//Explicitly instantiate so wrappers can access.
template const char *ardCudaGrad<double>(double inputX[], double *randomFeats,
                double precompWeights[], int32_t *sigmaMap,
                double *sigmaVals, double *gradient, int dim0,
                int dim1, int numLengthscales, int numFreqs,
                double rbfNormConstant);
template const char *ardCudaGrad<float>(float inputX[], double *randomFeats,
                float precompWeights[], int32_t *sigmaMap,
                double *sigmaVals, double *gradient, int dim0,
                int dim1, int numLengthscales, int numFreqs,
                double rbfNormConstant);
