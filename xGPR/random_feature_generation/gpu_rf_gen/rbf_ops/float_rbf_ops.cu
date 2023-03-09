/*
* Contains specialized functions for generating random features for
* the RBF and related kernels. It makes use of the hadamard transform functions
* implemented under array_operations.h, so only the pieces specific
* to the kernel need to be implemented here.
*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../float_array_operations.h"
#include "float_rbf_ops.h"
#include <cuda_profiler_api.h>


#define DEFAULT_THREADS_PER_BLOCK 256
#define MAX_BASE_LEVEL_TRANSFORM 1024



//Performs an elementwise multiplication of a [c,M,P] array against the
//[N,M,P] input array or a [P] array against the [N,P] input array.
//Note that the last dimensions of these must be the
//same, and this function does not check this -- caller must check. Note that
//we mutiiply by the Hadamard normalization constant here.
__global__ void floatSpecMultByDiagRademMat(float *cArray, int8_t *rademArray,
			int numElementsPerRow, int numElements, float normConstant)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int rVal, position;
    
    position = j % numElementsPerRow;
    rVal = rademArray[position];
    if (j < numElements)
        cArray[j] = cArray[j] * rVal * normConstant;
}



//Performs the last step in the random feature generation for the
//RBF / MiniARD kernels.
__global__ void rbfFeatureGenLastStepFloats(float *cArray, double *outputArray,
            float *chiArr, int numFreqs, int inputElementsPerRow,
            int numElements, double normConstant)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int chiArrPosition, inputPosition, outputRow, outputPosition;
    float outputVal;

    chiArrPosition = j % numFreqs;
    outputRow = (j / numFreqs);
    inputPosition = outputRow * inputElementsPerRow + chiArrPosition;
    outputPosition = outputRow * 2 * numFreqs + chiArrPosition;

    outputVal = chiArr[chiArrPosition] * cArray[inputPosition];
    if (j < numElements)
    {
        outputArray[outputPosition] = normConstant * cosf(outputVal);
        outputArray[outputPosition + numFreqs] = normConstant * sinf(outputVal);
    }
}


//Performs the last step in gradient / feature generation for RBF (NOT ARD)
//kernels.
__global__ void rbfGradLastStepFloats(float *cArray, double *outputArray,
            float *chiArr, double *gradientArray, float sigma, int numFreqs,
            int inputElementsPerRow, int numElements, double normConstant)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int chiArrPosition, inputPosition, outputRow, outputPosition;
    float outputVal, sinVal, cosVal;

    chiArrPosition = j % numFreqs;
    outputRow = (j / numFreqs);
    inputPosition = outputRow * inputElementsPerRow + chiArrPosition;
    outputPosition = outputRow * 2 * numFreqs + chiArrPosition;

    outputVal = chiArr[chiArrPosition] * cArray[inputPosition];
    if (j < numElements)
    {
        cosVal = normConstant * cosf(outputVal * sigma);
        sinVal = normConstant * sinf(outputVal * sigma);
        outputArray[outputPosition] = cosVal;
        outputArray[outputPosition + numFreqs] = sinVal;
        gradientArray[outputPosition] = -outputVal * sinVal;
        gradientArray[outputPosition + numFreqs] = outputVal * cosVal;
    }
}



//Performs the first piece of the gradient calculation for ARD kernels
//only -- multiplying the input data by the precomputed weight matrix
//and summing over rows that correspond to specific lengthscales.
__global__ void ardFloatGradSetup(double *gradientArray,
        float *precomputedWeights, float *inputX, int32_t *sigmaMap,
        int dim1, int numSetupElements, int gradIncrement,
        int numFreqs, int numLengthscales){

    int i, sigmaLoc;
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    //TODO: We are using multiple mod & integer divisions here --
    //find a more efficient way to implement this, mod and integer
    //division are expensive on GPU. Unfortunately these #s are not
    //necessarily powers of 2 in the current implementation.
    int precompWRow = (j % numFreqs);
    int gradRow = (j / numFreqs) / numFreqs;

    float *precompWElement = precomputedWeights + precompWRow * dim1;
    float *inputXElement = inputX + gradRow * dim1;
    double *gradientElement = gradientArray + (gradRow * numFreqs + precompWRow) * numLengthscales;
    float outVal;

    if (j < numSetupElements){
        for (i=0; i < dim1; i++){
            sigmaLoc = sigmaMap[i];
            outVal = precompWElement[i] * inputXElement[i];
            gradientElement[sigmaLoc] -= outVal;
            gradientElement[sigmaLoc + gradIncrement] += outVal;
        }
    }
}





//Multiplies the gradient array by the appropriate elements of the random
//feature array when calculating the gradient for ARD kernels only.
__global__ void ardFloatGradRFMultiply(double *gradientArray, double *randomFeats,
        int numRFElements, int numFreqs, int gradIncrement,
        int numLengthscales)
{
    int i;
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int rowNum = j / numFreqs, colNum = j % numFreqs;
    int gradPosition = rowNum * gradIncrement * 2;
    int rfPosition = rowNum * numFreqs * 2 + colNum;
    double rfVal, rfOffsetVal;
    

    if (j < numRFElements){
        rfVal = -randomFeats[rfPosition];
        rfOffsetVal = randomFeats[rfPosition + numFreqs];

        for (i=0; i < numLengthscales; i++)
            gradientArray[gradPosition + i] *= rfOffsetVal;

        gradPosition += gradIncrement;
        for (i=0; i < numLengthscales; i++)
            gradientArray[gradPosition + i] *= rfVal;
    }
}




//This function generates random features for RBF / ARD kernels, if the
//input has already been multiplied by the appropriate lengthscale values.
const char *floatRBFFeatureGen(float *cArray, int8_t *radem,
                float *chiArr, double *outputArray,
                double rbfNormConstant,
                int dim0, int dim1, int dim2,
                int numFreqs){
    int numElementsPerRow = dim1 * dim2;
    int numElements = dim1 * dim2 * dim0;
    //This is the Hadamard normalization constant.
    float normConstant = log2(dim2) / 2;
    normConstant = 1 / pow(2, normConstant);
    int blocksPerGrid = (numElements + DEFAULT_THREADS_PER_BLOCK - 1) / DEFAULT_THREADS_PER_BLOCK;
    int numOutputElements = numFreqs * dim0;
    //cudaProfilerStart();

    //Multiply by D1.
    floatSpecMultByDiagRademMat<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(cArray, radem, 
                                 numElementsPerRow, numElements, normConstant);
    
    //First H-transform.
    floatCudaHTransform3d(cArray, dim0, dim1, dim2);
    
    //Multiply by D2.
    floatSpecMultByDiagRademMat<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(cArray, radem + numElementsPerRow,
                                 numElementsPerRow, numElements, normConstant);

    //Second H-transform.
    floatCudaHTransform3d(cArray, dim0, dim1, dim2);
    
    //Multiply by D3.
    floatSpecMultByDiagRademMat<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(cArray, radem + 2 * numElementsPerRow,
                                 numElementsPerRow, numElements, normConstant);
    
    //Last H-transform.
    floatCudaHTransform3d(cArray, dim0, dim1, dim2); 


    //Generate output features in-place in the output array.
    blocksPerGrid = (numOutputElements + DEFAULT_THREADS_PER_BLOCK - 1) / DEFAULT_THREADS_PER_BLOCK;
    rbfFeatureGenLastStepFloats<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(cArray, outputArray,
                    chiArr, numFreqs, numElementsPerRow, numOutputElements, rbfNormConstant);

    //cudaProfilerStop();
    return "no_error";
}



//This function generates random features for RBF kernels ONLY
//(NOT ARD), and simultaneously generates the gradient, storing
//it in a separate array.
const char *floatRBFFeatureGrad(float *cArray, int8_t *radem,
                float *chiArr, double *outputArray,
                double *gradientArray, double rbfNormConstant,
                float sigma, int dim0, int dim1, int dim2,
                int numFreqs){
    int numElementsPerRow = dim1 * dim2;
    int numElements = dim1 * dim2 * dim0;
    //This is the Hadamard normalization constant.
    float normConstant = log2(dim2) / 2;
    normConstant = 1 / pow(2, normConstant);
    int blocksPerGrid = (numElements + DEFAULT_THREADS_PER_BLOCK - 1) / DEFAULT_THREADS_PER_BLOCK;
    int numOutputElements = numFreqs * dim0;
    //cudaProfilerStart();

    //Multiply by D1.
    floatSpecMultByDiagRademMat<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(cArray, radem, 
                                 numElementsPerRow, numElements, normConstant);
    
    //First H-transform.
    floatCudaHTransform3d(cArray, dim0, dim1, dim2);
    
    //Multiply by D2.
    floatSpecMultByDiagRademMat<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(cArray, radem + numElementsPerRow,
                                 numElementsPerRow, numElements, normConstant);

    //Second H-transform.
    floatCudaHTransform3d(cArray, dim0, dim1, dim2);
    
    //Multiply by D3.
    floatSpecMultByDiagRademMat<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(cArray, radem + 2 * numElementsPerRow,
                                 numElementsPerRow, numElements, normConstant);
    
    //Last H-transform.
    floatCudaHTransform3d(cArray, dim0, dim1, dim2); 


    //Generate output features in-place in the output array.
    blocksPerGrid = (numOutputElements + DEFAULT_THREADS_PER_BLOCK - 1) / DEFAULT_THREADS_PER_BLOCK;
    rbfGradLastStepFloats<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(cArray, outputArray,
                    chiArr, gradientArray, sigma, numFreqs,
                    numElementsPerRow, numOutputElements, rbfNormConstant);

    //cudaProfilerStop();
    return "no_error";
}



//This function generates the gradient ONLY for ARD ONLY,
//using random features that have already been generated
//and precomputed weights that take the place of the H-transforms
//we would otherwise need to perform.
const char *ardCudaFloatGrad(float *inputX, double *randomFeats,
                float *precompWeights, int32_t *sigmaMap,
                double *gradient, int dim0, int dim1,
                int numLengthscales, int numFreqs){

    int numRFElements = dim0 * numFreqs;
    int gradIncrement = numFreqs * numLengthscales;
    int numPrecompW = dim1 * numFreqs;
    int numSetupElements = numPrecompW;
    int blocksPerGrid;


    blocksPerGrid = (numSetupElements + DEFAULT_THREADS_PER_BLOCK - 1) / DEFAULT_THREADS_PER_BLOCK;
    ardFloatGradSetup<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(gradient, precompWeights, inputX,
            sigmaMap, dim1, numSetupElements, gradIncrement, numFreqs, numLengthscales);

    blocksPerGrid = (numRFElements + DEFAULT_THREADS_PER_BLOCK - 1) / DEFAULT_THREADS_PER_BLOCK;
    ardFloatGradRFMultiply<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(gradient, randomFeats,
                numRFElements, numFreqs, gradIncrement, numLengthscales);

    return "no_error";
}
