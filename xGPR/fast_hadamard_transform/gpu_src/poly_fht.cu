/*
* Contains functions needed to run FHT for the polynomial kernel on GPU.
* The input array should already live on GPU.
* The Hadamard transforms are performed using functions from float or double
* array_operations.cu, the diagonal matrix multiplication is slightly different
* and so is implemented here.
*/
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "double_array_operations.h"
#include "float_array_operations.h"
#include "poly_fht.h"

#define DEFAULT_THREADS_PER_BLOCK 256



//Performs an elementwise multiplication of a row of a [a,1,P x S] array against the
//[N,M,S] input array. Note that the dimensions must be checked before calling
//-- done by the wrapper -- and that only S elements of the [a, 1, P x S] array are used.
__global__ void floatPolyConvFHTMultiplyByRadem(float *cArray, int8_t *rademArray,
			int dim2, int columnStartPosition, int numElements, float normConstant)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int8_t *rVal = rademArray + columnStartPosition + (j & (dim2 - 1));
    
    if (j < numElements)
        cArray[j] = cArray[j] * *rVal * normConstant;
}



//Performs an elementwise multiplication of a row of a [a,1,P x S] array against the
//[N,M,S] input array. Note that the dimensions must be checked before calling
//-- done by the wrapper -- and that only S elements of the [a, 1, P x S] array are used.
__global__ void doublePolyConvFHTMultiplyByRadem(double *cArray, int8_t *rademArray,
			int dim2, int columnStartPosition, int numElements, double normConstant)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int8_t *rVal = rademArray + columnStartPosition + (j & (dim2 - 1));
    
    if (j < numElements)
        cArray[j] = cArray[j] * *rVal * normConstant;
}



//Performs an elementwise multiplication of a row of a [a, M, S] array against the
//[N,M,S] input array. Note that the dimensions must be checked before calling
//-- done by the wrapper -- and that only S elements of the [a, M, S] array are used.
__global__ void floatPolyFHTMultiplyByRadem(float *cArray, int8_t *rademArray,
			int numElementsPerRow, int numElements, float normConstant)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int rVal, position;
    
    position = j % numElementsPerRow;
    rVal = rademArray[position];
    if (j < numElements)
        cArray[j] *= rVal * normConstant;
}




//Performs an elementwise multiplication of a row of a [a, M, S] array against the
//[N,M,S] input array. Note that the dimensions must be checked before calling
//-- done by the wrapper -- and that only S elements of the [a, M, S] array are used.
__global__ void doublePolyFHTMultiplyByRadem(double *cArray, int8_t *rademArray,
			int numElementsPerRow, int numElements, double normConstant)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int rVal, position;
    
    position = j % numElementsPerRow;
    rVal = rademArray[position];
    if (j < numElements)
        cArray[j] *= rVal * normConstant;
}





//This function performs the FHT operation for the polynomial kernel
//when the input is an arrray of floats and when graph convolution
//is desired.
//Note that reshapedX must have the same size across the
//last two dimensions as radem and its last dimension must
//be a power of two -- if those conditions are not met, you may
//get an unpredictable result! The Cython wrapper checks all
//of these criteria.
const char *floatPolyConvFHTPrep(int8_t *radem, float *reshapedX, int reshapedDim0, 
                int reshapedDim1, int reshapedDim2, int numFreqs,
                int columnStartPosition, int rowStartPosition){

    int numElements = reshapedDim0 * reshapedDim1 * reshapedDim2;
    int rowOffset;
    //This is the Hadamard normalization constant.
    float normConstant = log2(reshapedDim2) / 2;
    normConstant = 1 / pow(2, normConstant);
    int blocksPerGrid = (numElements + DEFAULT_THREADS_PER_BLOCK - 1) / 
                DEFAULT_THREADS_PER_BLOCK;
    
    //Multiply by radem row 1.
    rowOffset = rowStartPosition * numFreqs;
    floatPolyConvFHTMultiplyByRadem<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(reshapedX, 
                        radem + rowOffset, reshapedDim2, columnStartPosition, numElements,
                        normConstant);
    //First H-transform.
    floatCudaHTransform3d(reshapedX, reshapedDim0, reshapedDim1, reshapedDim2);
    //All operations are in place, no need to return anything except a 
    //no error message. TODO: check the cuda kernels for errors and add error
    //handling.
    return "no_error";
}



//This function performs the FHT operation for the polynomial kernel
//when the input is an arrray of doubles and graph convolution is
//desired.
//Note that reshapedX must have the same size across the
//last two dimensions as radem and its last dimension must
//be a power of two -- if those conditions are not met, you may
//get an unpredictable result! The Cython wrapper checks all
//of these criteria.
const char *doublePolyConvFHTPrep(int8_t *radem, double *reshapedX, int reshapedDim0, 
                int reshapedDim1, int reshapedDim2, int numFreqs,
                int columnStartPosition, int rowStartPosition){

    int numElements = reshapedDim0 * reshapedDim1 * reshapedDim2;
    int rowOffset;
    //This is the Hadamard normalization constant.
    double normConstant = log2(reshapedDim2) / 2;
    normConstant = 1 / pow(2, normConstant);
    int blocksPerGrid = (numElements + DEFAULT_THREADS_PER_BLOCK - 1) / 
                DEFAULT_THREADS_PER_BLOCK;
    
    //Multiply by radem row 1.
    rowOffset = rowStartPosition * numFreqs;
    doublePolyConvFHTMultiplyByRadem<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(reshapedX, 
                        radem + rowOffset, reshapedDim2, columnStartPosition, numElements,
                        normConstant);
    //First H-transform.
    doubleCudaHTransform3d(reshapedX, reshapedDim0, reshapedDim1, reshapedDim2);
    //All operations are in place, no need to return anything except a 
    //no error message. TODO: check the cuda kernels for errors and add error
    //handling.
    return "no_error";
}




//This function performs the FHT operation for the polynomial kernel
//when the input is an arrray of floats and the input is fixed
//vector data.
//Note that reshapedX must have the same size across the
//last two dimensions as radem and its last dimension must
//be a power of two -- if those conditions are not met, you may
//get an unpredictable result! The Cython wrapper checks all
//of these criteria.
const char *floatPolyFHTPrep(int8_t *radem, float *reshapedX, int reshapedDim0, 
                int reshapedDim1, int reshapedDim2, int rademStartPosition){

    int numElements = reshapedDim0 * reshapedDim1 * reshapedDim2;
    int numElementsPerRow = reshapedDim1 * reshapedDim2;
    //This is the Hadamard normalization constant.
    float normConstant = log2(reshapedDim2) / 2;
    normConstant = 1 / pow(2, normConstant);
    int blocksPerGrid = (numElements + DEFAULT_THREADS_PER_BLOCK - 1) / 
                DEFAULT_THREADS_PER_BLOCK;
    
    //Multiply by radem.
    floatPolyFHTMultiplyByRadem<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(reshapedX, 
                        radem + rademStartPosition, numElementsPerRow, numElements, normConstant);
    //First H-transform.
    floatCudaHTransform3d(reshapedX, reshapedDim0, reshapedDim1, reshapedDim2);
    //All operations are in place, no need to return anything except a 
    //no error message. TODO: check the cuda kernels for errors and add error
    //handling.
    return "no_error";
}



//This function performs the FHT operation for the polynomial kernel
//when the input is an arrray of doubles and graph convolution is
//desired.
//Note that reshapedX must have the same size across the
//last two dimensions as radem and its last dimension must
//be a power of two -- if those conditions are not met, you may
//get an unpredictable result! The Cython wrapper checks all
//of these criteria.
const char *doublePolyFHTPrep(int8_t *radem, double *reshapedX, int reshapedDim0, 
                int reshapedDim1, int reshapedDim2, int rademStartPosition){

    int numElements = reshapedDim0 * reshapedDim1 * reshapedDim2;
    int numElementsPerRow = reshapedDim1 * reshapedDim2;
    //This is the Hadamard normalization constant.
    double normConstant = log2(reshapedDim2) / 2;
    normConstant = 1 / pow(2, normConstant);
    int blocksPerGrid = (numElements + DEFAULT_THREADS_PER_BLOCK - 1) / 
                DEFAULT_THREADS_PER_BLOCK;
    
    //Multiply by radem row 1.
    doublePolyFHTMultiplyByRadem<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(reshapedX, 
                        radem + rademStartPosition, numElementsPerRow, numElements, normConstant);
    //First H-transform.
    doubleCudaHTransform3d(reshapedX, reshapedDim0, reshapedDim1, reshapedDim2);
    //All operations are in place, no need to return anything except a 
    //no error message. TODO: check the cuda kernels for errors and add error
    //handling.
    return "no_error";
}
