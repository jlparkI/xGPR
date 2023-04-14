/*
* Contains specialized functions for generating random features for
* the RBF and related kernels. It makes use of the hadamard transform functions
* implemented under array_operations.h, so only the pieces specific
* to the kernel need to be implemented here.
*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#include "../basic_ops/double_array_operations.h"
#include "double_rbf_ops.h"


#define DEFAULT_THREADS_PER_BLOCK 256
#define MAX_BASE_LEVEL_TRANSFORM 1024

//Performs an elementwise multiplication of a [c,M,P] array against the
//[N,M,P] input array or a [P] array against the [N,P] input array.
//Note that the last dimensions of these must be the
//same, and this function does not check this -- caller must check. Note that
//we mutiiply by the Hadamard normalization constant here.
__global__ void doubleSpecMultByDiagRademMat(double *cArray, int8_t *rademArray,
			int numElementsPerRow, int numElements, double normConstant)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int rVal, position;
    
    position = tid % numElementsPerRow;
    rVal = rademArray[position];
    if (tid < numElements)
        cArray[tid] = cArray[tid] * rVal * normConstant;
}



//Performs the last step in the random feature generation for the
//RBF / MiniARD kernels.
__global__ void rbfFeatureGenLastStepDoubles(double *cArray, double *outputArray,
            double *chiArr, int numFreqs, int inputElementsPerRow,
            int numElements, double normConstant)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int chiArrPosition, inputPosition, outputRow, outputPosition;
    double outputVal;

    chiArrPosition = tid % numFreqs;
    outputRow = (tid / numFreqs);
    inputPosition = outputRow * inputElementsPerRow + chiArrPosition;
    //Multiply by 2 here since we store both the sine and cosine
    //of the feature in the output array.
    outputPosition = 2 * (outputRow * numFreqs + chiArrPosition);

    outputVal = chiArr[chiArrPosition] * cArray[inputPosition];
    if (tid < numElements)
    {
        outputArray[outputPosition] = normConstant * cos(outputVal);
        outputArray[outputPosition + 1] = normConstant * sin(outputVal);
    }
}



//Performs the last step in gradient / feature generation for RBF (NOT ARD)
//kernels.
__global__ void rbfGradLastStepDoubles(double *cArray, double *outputArray,
            double *chiArr, double *gradientArray, double sigma, int numFreqs,
            int inputElementsPerRow, int numElements, double normConstant)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int chiArrPosition, inputPosition, outputRow, outputPosition;
    double outputVal, sinVal, cosVal;

    chiArrPosition = tid % numFreqs;
    outputRow = (tid / numFreqs);
    inputPosition = outputRow * inputElementsPerRow + chiArrPosition;
    //Multiply by 2 here since we store both the sine and cosine
    //of the feature in the output array.
    outputPosition = 2 * (outputRow * numFreqs + chiArrPosition);

    outputVal = chiArr[chiArrPosition] * cArray[inputPosition];
    if (tid < numElements)
    {
        cosVal = normConstant * cos(outputVal * sigma);
        sinVal = normConstant * sin(outputVal * sigma);
        outputArray[outputPosition] = cosVal;
        outputArray[outputPosition + 1] = sinVal;
        gradientArray[outputPosition] = -outputVal * sinVal;
        gradientArray[outputPosition + 1] = outputVal * cosVal;
    }
}



//Performs the first piece of the gradient calculation for ARD kernels
//only -- multiplying the input data by the precomputed weight matrix
//and summing over rows that correspond to specific lengthscales.
__global__ void ardDoubleGradSetup(double *gradientArray,
        double *precomputedWeights, double *inputX, int32_t *sigmaMap,
        double *sigmaVals, double *randomFeatures,
        int dim1, int numSetupElements, int numFreqs,
        int numLengthscales){

    int i, sigmaLoc;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int precompWRow = (tid % numFreqs);
    int gradRow = tid / numFreqs;

    double *precompWElement = precomputedWeights + precompWRow * dim1;
    double *inputXElement = inputX + gradRow * dim1;
    double *gradientElement = gradientArray + 2 * (gradRow * numFreqs + precompWRow) * numLengthscales;
    double *randomFeature = randomFeatures + 2 * (gradRow * numFreqs + precompWRow);
    double rfVal = 0;
    double outVal;

    if (tid < numSetupElements){
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
__global__ void ardDoubleGradRFMultiply(double *gradientArray, double *randomFeats,
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




//This function generates random features for RBF / ARD kernels, if the
//input has already been multiplied by the appropriate lengthscale values.
const char *doubleRBFFeatureGen(double *cArray, int8_t *radem,
                double *chiArr, double *outputArray,
                double rbfNormConstant,
                int dim0, int dim1, int dim2,
                int numFreqs){
    int numElementsPerRow = dim1 * dim2;
    int numElements = dim1 * dim2 * dim0;
    //This is the Hadamard normalization constant.
    double normConstant = log2(dim2) / 2;
    normConstant = 1 / pow(2, normConstant);
    int blocksPerGrid = (numElements + DEFAULT_THREADS_PER_BLOCK - 1) / DEFAULT_THREADS_PER_BLOCK;
    int numOutputElements = numFreqs * dim0;

    //Multiply by D1.
    doubleSpecMultByDiagRademMat<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(cArray, radem, 
                                 numElementsPerRow, numElements, normConstant);
    
    //First H-transform.
    doubleCudaHTransform3d(cArray, dim0, dim1, dim2);
    
    //Multiply by D2.
    doubleSpecMultByDiagRademMat<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(cArray, radem + numElementsPerRow,
                                 numElementsPerRow, numElements, normConstant);

    //Second H-transform.
    doubleCudaHTransform3d(cArray, dim0, dim1, dim2);
    
    //Multiply by D3.
    doubleSpecMultByDiagRademMat<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(cArray, radem + 2 * numElementsPerRow,
                                 numElementsPerRow, numElements, normConstant);
    
    //Last H-transform.
    doubleCudaHTransform3d(cArray, dim0, dim1, dim2); 


    //Generate output features in-place in the output array.
    blocksPerGrid = (numOutputElements + DEFAULT_THREADS_PER_BLOCK - 1) / DEFAULT_THREADS_PER_BLOCK;
    rbfFeatureGenLastStepDoubles<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(cArray, outputArray,
                    chiArr, numFreqs, numElementsPerRow, numOutputElements, rbfNormConstant);

    return "no_error";
}



//This function generates random features for RBF kernels ONLY
//(NOT ARD), and simultaneously generates the gradient, storing
//it in a separate array.
const char *doubleRBFFeatureGrad(double *cArray, int8_t *radem,
                double *chiArr, double *outputArray,
                double *gradientArray, double rbfNormConstant,
                double sigma, int dim0, int dim1, int dim2,
                int numFreqs){

    int numElementsPerRow = dim1 * dim2;
    int numElements = dim1 * dim2 * dim0;
    //This is the Hadamard normalization constant.
    double normConstant = log2(dim2) / 2;
    normConstant = 1 / pow(2, normConstant);
    int blocksPerGrid = (numElements + DEFAULT_THREADS_PER_BLOCK - 1) / DEFAULT_THREADS_PER_BLOCK;
    int numOutputElements = numFreqs * dim0;

    //Multiply by D1.
    doubleSpecMultByDiagRademMat<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(cArray, radem, 
                                 numElementsPerRow, numElements, normConstant);
    
    //First H-transform.
    doubleCudaHTransform3d(cArray, dim0, dim1, dim2);
    
    //Multiply by D2.
    doubleSpecMultByDiagRademMat<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(cArray, radem + numElementsPerRow,
                                 numElementsPerRow, numElements, normConstant);

    //Second H-transform.
    doubleCudaHTransform3d(cArray, dim0, dim1, dim2);
    
    //Multiply by D3.
    doubleSpecMultByDiagRademMat<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(cArray, radem + 2 * numElementsPerRow,
                                 numElementsPerRow, numElements, normConstant);
    
    //Last H-transform.
    doubleCudaHTransform3d(cArray, dim0, dim1, dim2); 


    //Generate output features in-place in the output array.
    blocksPerGrid = (numOutputElements + DEFAULT_THREADS_PER_BLOCK - 1) / DEFAULT_THREADS_PER_BLOCK;
    rbfGradLastStepDoubles<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(cArray, outputArray,
                    chiArr, gradientArray, sigma, numFreqs,
                    numElementsPerRow, numOutputElements, rbfNormConstant);

    return "no_error";
}



//This function generates the gradient and random features
//for ARD kernels only, using precomputed weights that take
//the place of the H-transforms
//we would otherwise need to perform.
const char *ardCudaDoubleGrad(double *inputX, double *randomFeats,
                double *precompWeights, int32_t *sigmaMap,
                double *sigmaVals, double *gradient, int dim0,
                int dim1, int numLengthscales, int numFreqs,
                double rbfNormConstant){

    int numRFElements = dim0 * numFreqs;
    int numSetupElements = dim0 * numFreqs;
    int blocksPerGrid;


    blocksPerGrid = (numSetupElements + DEFAULT_THREADS_PER_BLOCK - 1) / DEFAULT_THREADS_PER_BLOCK;
    ardDoubleGradSetup<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(gradient, precompWeights, inputX,
            sigmaMap, sigmaVals, randomFeats, dim1, numSetupElements,
            numFreqs, numLengthscales);

    blocksPerGrid = (numRFElements + DEFAULT_THREADS_PER_BLOCK - 1) / DEFAULT_THREADS_PER_BLOCK;
    ardDoubleGradRFMultiply<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(gradient, randomFeats,
                numRFElements, numFreqs, numLengthscales, rbfNormConstant);

    return "no_error";
}
