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
