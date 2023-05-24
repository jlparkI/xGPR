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





//Initializes all elements of the copy buffer to zero.
__global__ void initializeARDConvGradBuffer(double *copyBuffer,
        int numBufferElements){

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < numBufferElements)
        copyBuffer[tid] = 0;
}



//Performs the first piece of the gradient calculation for convolution
//ARD kernels only -- multiplying the input data by the precomputed
//weight matrix and summing over rows that correspond to
//specific lengthscales.
template <typename T>
__global__ void ardConvGradSetup(double *gradientArray,
        T precomputedWeights[], T inputX[], int32_t *sigmaMap,
        double *copyBuffer, double *sigmaVals, double *randomFeatures,
        int dim1, int dim2, int numSetupElements,
        int numFreqs, int numLengthscales,
        double rbfNormConstant){

    int i, j;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int precompWRow = (tid % numFreqs);
    int gradRow = tid / numFreqs;

    T *precompWElement = precomputedWeights + precompWRow * dim2;
    T *inputXElement = inputX + gradRow * dim1 * dim2;
    double *gradientElement = gradientArray + 2 * (gradRow * numFreqs + precompWRow) * numLengthscales;
    double *randomFeature = randomFeatures + 2 * gradRow * numFreqs + 2 * precompWRow;
    double *bufferElement = copyBuffer + (gradRow * numFreqs + precompWRow) * numLengthscales;
    double rfVal = 0, outVal, sinVal, cosVal;

    if (tid < numSetupElements){
        for (i=0; i < dim1; i++){
            rfVal = 0;
            for (j=0; j < dim2; j++){
                outVal = precompWElement[j] * inputXElement[j];
                bufferElement[sigmaMap[j]] += outVal;
                rfVal += sigmaVals[j] * outVal;
            }
            cosVal = rbfNormConstant * cos(rfVal);
            sinVal = rbfNormConstant * sin(rfVal);
            *randomFeature += cosVal;
            randomFeature[1] += sinVal;

            for (j=0; j < numLengthscales; j++){
                gradientElement[j] -= bufferElement[j] * sinVal;
                gradientElement[j + numLengthscales] += bufferElement[j] * cosVal;
                bufferElement[j] = 0;
            }
            inputXElement += dim2;
        }
    }
}



//This function generates the gradient and random features for ARD ONLY,
//using precomputed weights that take the place of the H-transforms
//we would otherwise need to perform.
const char *ardConvCudaFloatGrad(float *inputX, double *randomFeats,
                float *precompWeights, int32_t *sigmaMap,
                double *sigmaVals, double *gradient, int dim0,
                int dim1, int dim2, int numLengthscales,
                int numFreqs, double rbfNormConstant){

    int numRFElements = dim0 * numFreqs;
    int numBufferElements = dim0 * numFreqs * numLengthscales;
    int blocksPerGrid;
    double *copyBuffer;

    cudaMalloc(&copyBuffer, numBufferElements * sizeof(double)); 

    blocksPerGrid = (numBufferElements + DEFAULT_THREADS_PER_BLOCK - 1) / DEFAULT_THREADS_PER_BLOCK;
    
    initializeARDConvGradBuffer<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(copyBuffer,
                numBufferElements);

    blocksPerGrid = (numRFElements + DEFAULT_THREADS_PER_BLOCK - 1) / DEFAULT_THREADS_PER_BLOCK;

    ardConvGradSetup<float><<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(gradient, precompWeights, inputX,
            sigmaMap, copyBuffer, sigmaVals, randomFeats, dim1, dim2, numRFElements,
            numFreqs, numLengthscales, rbfNormConstant);

    cudaFree(copyBuffer);

    return "no_error";
}




//This function generates the gradient and random features for ARD ONLY,
//using precomputed weights that take the place of the H-transforms
//we would otherwise need to perform.
const char *ardConvCudaDoubleGrad(double *inputX, double *randomFeats,
                double *precompWeights, int32_t *sigmaMap,
                double *sigmaVals, double *gradient, int dim0,
                int dim1, int dim2, int numLengthscales,
                int numFreqs, double rbfNormConstant){

    int numRFElements = dim0 * numFreqs;
    int numBufferElements = dim0 * numFreqs * numLengthscales;
    int blocksPerGrid;
    double *copyBuffer;

    cudaMalloc(&copyBuffer, numBufferElements * sizeof(double)); 

    blocksPerGrid = (numBufferElements + DEFAULT_THREADS_PER_BLOCK - 1) / DEFAULT_THREADS_PER_BLOCK;
    
    initializeARDConvGradBuffer<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(copyBuffer,
                numBufferElements);

    blocksPerGrid = (numRFElements + DEFAULT_THREADS_PER_BLOCK - 1) / DEFAULT_THREADS_PER_BLOCK;

    ardConvGradSetup<double><<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(gradient,
            precompWeights, inputX, sigmaMap, copyBuffer, sigmaVals, randomFeats,
            dim1, dim2, numRFElements, numFreqs,
            numLengthscales, rbfNormConstant);

    cudaFree(copyBuffer);

    return "no_error";
}
