/*
* Contains all functions needed to run the structured orthogonal features
* (SORF) operation on an input 3d array that has been restructured for
* convolution on GPU. The input array should already live on GPU.
* The Hadamard transforms are performed using functions from float_array_operations.cu
* and double_array_operations.cu, the diagonal matrix multiplication is slightly
* different and so is
* implemented here.
*/
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "float_array_operations.h"
#include "double_array_operations.h"
#include "convolution.h"

#define DEFAULT_THREADS_PER_BLOCK 256



//Performs an elementwise multiplication of a row of a [3,1,P x S] array against the
//float [N,M,S] input array. Note that the dimensions must be checked before calling
//-- done by the wrapper -- and that only S elements of the appropriate row of
//the [3, 1, P x S] array are used.
__global__ void floatConv1dMultiplyByRadem(float *cArray, int8_t *rademArray,
			int dim2, int startPosition, int numElements, float normConstant)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int8_t *rVal = rademArray + startPosition + (j & (dim2 - 1));
    
    if (j < numElements)
        cArray[j] = cArray[j] * *rVal * normConstant;
}



//Performs an elementwise multiplication of a row of a [3,1,P x S] array against the
//double [N,M,S] input array. Note that the dimensions must be checked before calling
//-- done by the wrapper -- and that only S elements of the appropriate row of
//the [3, 1, P x S] array are used.
__global__ void doubleConv1dMultiplyByRadem(double *cArray, int8_t *rademArray,
			int dim2, int startPosition, int numElements, double normConstant)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int8_t *rVal = rademArray + startPosition + (j & (dim2 - 1));
    
    if (j < numElements)
        cArray[j] = cArray[j] * *rVal * normConstant;
}


//Performs an elementwise multiplication of a row of a [3,1,P x S] array against the
//float [N,M,S] input array. Note that the dimensions must be checked before calling
//-- done by the wrapper -- and that only S elements of the appropriate row of
//the [3, 1, P x S] array are used. In this case (unlike for MultiplyByRadem)
//the diagonal matrix has floats along the diagonal rather than int8.
__global__ void floatConv1dMultiplyByDiag(float *cArray, float *rademArray,
			int dim2, int startPosition, int numElements, float normConstant)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    float *rVal = rademArray + startPosition + (j & (dim2 - 1));
    
    if (j < numElements)
        cArray[j] = cArray[j] * *rVal * normConstant;
}



//Performs an elementwise multiplication of a row of a [3,1,P x S] array against the
//double [N,M,S] input array. Note that the dimensions must be checked before calling
//-- done by the wrapper -- and that only S elements of the appropriate row of
//the [3, 1, P x S] array are used. In this case (unlike for MultiplyByRadem)
//the diagonal matrix has doubles along the diagonal rather than int8.
__global__ void doubleConv1dMultiplyByDiag(double *cArray, double *rademArray,
			int dim2, int startPosition, int numElements, double normConstant)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    double *rVal = rademArray + startPosition + (j & (dim2 - 1));
    
    if (j < numElements)
        cArray[j] = cArray[j] * *rVal * normConstant;
}


//This function performs the SORF block transform (HD3 HD2 HD1)
//on float input that has been reshaped as appropriate for a convolution.
//Note that reshapedX must have the same size across the
//last two dimensions as radem and its last dimension must
//be a power of two -- if those conditions are not met, you may
//get an unpredictable result! The Cython wrapper checks all
//of these criteria.
const char *floatConv1dPrep(int8_t *radem, float *reshapedX, int reshapedDim0, 
                int reshapedDim1, int reshapedDim2, int startPosition,
                int numFreqs){

    int numElements = reshapedDim0 * reshapedDim1 * reshapedDim2;
    //This is the Hadamard normalization constant.
    float normConstant = log2(reshapedDim2) / 2;
    normConstant = 1 / pow(2, normConstant);
    int blocksPerGrid = (numElements + DEFAULT_THREADS_PER_BLOCK - 1) / 
                DEFAULT_THREADS_PER_BLOCK;
    
    //Multiply by first row of radem.
    floatConv1dMultiplyByRadem<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(reshapedX, 
                        radem, reshapedDim2, startPosition, numElements,
                        normConstant);
    //First H-transform.
    floatCudaHTransform3d(reshapedX, reshapedDim0, reshapedDim1, reshapedDim2);

    //Multiply by second row of radem.
    floatConv1dMultiplyByRadem<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(reshapedX, 
                        radem + numFreqs, reshapedDim2, startPosition, numElements,
                        normConstant);
    //Second H-transform.
    floatCudaHTransform3d(reshapedX, reshapedDim0, reshapedDim1, reshapedDim2);
        
    //Multiply by third row of radem.
    floatConv1dMultiplyByRadem<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(reshapedX, 
                        radem + 2 * numFreqs, reshapedDim2, startPosition, numElements,
                        normConstant);
    //Last H-transform. Transform is in place so do not need to return anything except no error message.
    floatCudaHTransform3d(reshapedX, reshapedDim0, reshapedDim1, reshapedDim2);
    return "no_error";
}



//This function performs the SORF block transform (HD3 HD2 HD1)
//on double input that has been reshaped as appropriate for a convolution.
//Note that reshapedX must have the same size across the
//last two dimensions as radem and its last dimension must
//be a power of two -- if those conditions are not met, you may
//get an unpredictable result! The Cython wrapper checks all
//of these criteria.
const char *doubleConv1dPrep(int8_t *radem, double *reshapedX, int reshapedDim0, 
                int reshapedDim1, int reshapedDim2, int startPosition,
                int numFreqs){

    int numElements = reshapedDim0 * reshapedDim1 * reshapedDim2;
    //This is the Hadamard normalization constant.
    double normConstant = log2(reshapedDim2) / 2;
    normConstant = 1 / pow(2, normConstant);
    int blocksPerGrid = (numElements + DEFAULT_THREADS_PER_BLOCK - 1) / 
                DEFAULT_THREADS_PER_BLOCK;
    
    //Multiply by first row of radem.
    doubleConv1dMultiplyByRadem<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(reshapedX, 
                        radem, reshapedDim2, startPosition, numElements,
                        normConstant);
    //First H-transform.
    doubleCudaHTransform3d(reshapedX, reshapedDim0, reshapedDim1, reshapedDim2);

    //Multiply by second row of radem.
    doubleConv1dMultiplyByRadem<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(reshapedX, 
                        radem + numFreqs, reshapedDim2, startPosition, numElements,
                        normConstant);
    //Second H-transform.
    doubleCudaHTransform3d(reshapedX, reshapedDim0, reshapedDim1, reshapedDim2);
        
    //Multiply by third row of radem.
    doubleConv1dMultiplyByRadem<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(reshapedX, 
                        radem + 2 * numFreqs, reshapedDim2, startPosition, numElements,
                        normConstant);
    //Last H-transform. Transform is in place so do not need to return anything except no error message.
    doubleCudaHTransform3d(reshapedX, reshapedDim0, reshapedDim1, reshapedDim2);
    return "no_error";
}
