/*
* Contains routines needed specifically for generating features and gradients
* for RBF-based convolution ARD kernels.
*/
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#include "ard_convolution.h"

#define DEFAULT_THREADS_PER_BLOCK 256
#define DEFAULT_THREADS_PER_BLREDUCE 32



//Performs the first piece of the gradient calculation for convolution
//ARD kernels only -- multiplying the input data by the precomputed
//weight matrix and summing over rows that correspond to
//specific lengthscales.
__global__ void ardConvDoubleGradSetup(double *gradientArray,
        double *precomputedWeights, double *inputX, int32_t *sigmaMap,
        double *sigmaVals, double *randomFeatures,
        int dim1, int dim2, int numSetupElements,
        int gradIncrement, int numFreqs,
        int numLengthscales){

    int i, j, sigmaLoc;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int precompWRow = (tid % numFreqs);
    int gradRow = tid / numFreqs;

    double *precompWElement = precomputedWeights + precompWRow * dim2;
    double *inputXElement = inputX + gradRow * dim1 * dim2;
    double *gradientElement = gradientArray + (gradRow * numFreqs * 2 + precompWRow) * numLengthscales;
    double *randomFeature = randomFeatures + gradRow * numFreqs * 2 + precompWRow;
    double rfVal = 0, outVal;

    if (tid < numSetupElements){
        for (i=0; i < dim1; i++){
            rfVal = 0;
            for (j=0; j < dim2; j++){
                sigmaLoc = sigmaMap[j];
                outVal = precompWElement[j] * inputXElement[j];
                gradientElement[sigmaLoc] += outVal;
                rfVal += sigmaVals[j] * outVal;
            }
            *randomFeature += cos(rfVal);
            randomFeature[numFreqs] += sin(rfVal);
            inputXElement += dim2;
        }
    }
}



//Multiplies the gradient array by the appropriate elements of the random
//feature array when calculating the gradient.
__global__ void ardConvDoubleGradRFMultiply(double *gradientArray, double *randomFeats,
        int numRFElements, int numFreqs, int gradIncrement,
        int numLengthscales, double rbfNormConstant){
    int i;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int rowNum = tid / numFreqs, colNum = tid % numFreqs;
    int gradPosition = rowNum * gradIncrement * 2 + colNum * numLengthscales;
    int rfPosition = rowNum * numFreqs * 2 + colNum;
    double rfVal, cosVal, sinVal;
    

    if (tid < numRFElements){
        cosVal = randomFeats[rfPosition] * rbfNormConstant;
        sinVal = randomFeats[rfPosition + numFreqs] * rbfNormConstant;
        randomFeats[rfPosition] = cosVal;
        randomFeats[rfPosition + numFreqs] = sinVal;

        for (i=0; i < numLengthscales; i++){
            rfVal = gradientArray[gradPosition + i];
            gradientArray[gradPosition + i] = -rfVal * sinVal;
            gradientArray[gradPosition + i + gradIncrement] = rfVal * cosVal;
        }
    }
}



//Performs the first piece of the gradient calculation for convolution
//ARD kernels only -- multiplying the input data by the precomputed
//weight matrix and summing over rows that correspond to
//specific lengthscales.
__global__ void ardConvFloatGradSetup(double *gradientArray,
        float *precomputedWeights, float *inputX, int32_t *sigmaMap,
        double *sigmaVals, double *randomFeatures,
        int dim1, int dim2, int numSetupElements,
        int gradIncrement, int numFreqs,
        int numLengthscales){

    int i, j, sigmaLoc;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int precompWRow = (tid % numFreqs);
    int gradRow = tid / numFreqs;

    float *precompWElement = precomputedWeights + precompWRow * dim2;
    float *inputXElement = inputX + gradRow * dim1 * dim2;
    double *gradientElement = gradientArray + (gradRow * numFreqs * 2 + precompWRow) * numLengthscales;
    double *randomFeature = randomFeatures + gradRow * numFreqs * 2 + precompWRow;
    double rfVal;
    float outVal;

    if (tid < numSetupElements){
        for (i=0; i < dim1; i++){
            rfVal = 0;
            for (j=0; j < dim2; j++){
                sigmaLoc = sigmaMap[j];
                outVal = precompWElement[j] * inputXElement[j];
                gradientElement[sigmaLoc] += outVal;
                rfVal += sigmaVals[j] * outVal;
            }
            inputXElement += dim2;
            *randomFeature += cos(rfVal);
            randomFeature[numFreqs] += sin(rfVal);
        }
    }
}




//Multiplies the gradient array by the appropriate elements of the random
//feature array when calculating the gradient.
__global__ void ardConvFloatGradRFMultiply(double *gradientArray, double *randomFeats,
        int numRFElements, int numFreqs, int gradIncrement,
        int numLengthscales, double rbfNormConstant){
    int i;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int rowNum = tid / numFreqs, colNum = tid % numFreqs;
    int gradPosition = rowNum * gradIncrement * 2 + colNum * numLengthscales;
    int rfPosition = rowNum * numFreqs * 2 + colNum;
    double rfVal, cosVal, sinVal;
    

    if (tid < numRFElements){
        cosVal = randomFeats[rfPosition] * rbfNormConstant;
        sinVal = randomFeats[rfPosition + numFreqs] * rbfNormConstant;
        randomFeats[rfPosition] = cosVal;
        randomFeats[rfPosition + numFreqs] = sinVal;

        for (i=0; i < numLengthscales; i++){
            rfVal = gradientArray[gradPosition + i];
            gradientArray[gradPosition + i] = -rfVal * sinVal;
            gradientArray[gradPosition + i + gradIncrement] = rfVal * cosVal;
        }
    }
}



//This function generates the gradient ONLY for ARD ONLY,
//using random features that have already been generated
//and precomputed weights that take the place of the H-transforms
//we would otherwise need to perform.
const char *ardConvCudaFloatGrad(float *inputX, double *randomFeats,
                float *precompWeights, int32_t *sigmaMap,
                double *sigmaVals, double *gradient, int dim0,
                int dim1, int dim2, int numLengthscales,
                int numFreqs, double rbfNormConstant){

    int numRFElements = dim0 * numFreqs;
    int gradIncrement = numFreqs * numLengthscales;
    int blocksPerGrid;


    blocksPerGrid = (numRFElements + DEFAULT_THREADS_PER_BLOCK - 1) / DEFAULT_THREADS_PER_BLOCK;
    ardConvFloatGradSetup<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(gradient, precompWeights, inputX,
            sigmaMap, sigmaVals, randomFeats, dim1, dim2, numRFElements,
            gradIncrement, numFreqs, numLengthscales);

    ardConvFloatGradRFMultiply<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(gradient, randomFeats,
                numRFElements, numFreqs, gradIncrement, numLengthscales, rbfNormConstant);

    return "no_error";
}




//This function generates the gradient ONLY for ARD ONLY,
//using random features that have already been generated
//and precomputed weights that take the place of the H-transforms
//we would otherwise need to perform.
const char *ardConvCudaDoubleGrad(double *inputX, double *randomFeats,
                double *precompWeights, int32_t *sigmaMap,
                double *sigmaVals, double *gradient, int dim0,
                int dim1, int dim2, int numLengthscales,
                int numFreqs, double rbfNormConstant){

    int numRFElements = dim0 * numFreqs;
    int gradIncrement = numFreqs * numLengthscales;
    int blocksPerGrid;


    blocksPerGrid = (numRFElements + DEFAULT_THREADS_PER_BLOCK - 1) / DEFAULT_THREADS_PER_BLOCK;

    ardConvDoubleGradSetup<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(gradient,
            precompWeights, inputX, sigmaMap, sigmaVals, randomFeats,
            dim1, dim2, numRFElements, gradIncrement,
            numFreqs, numLengthscales);

    ardConvDoubleGradRFMultiply<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(gradient,
            randomFeats, numRFElements, numFreqs, gradIncrement,
            numLengthscales, rbfNormConstant);

    return "no_error";
}
