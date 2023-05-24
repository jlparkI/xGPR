/*
* Contains all functions needed to run the structured orthogonal features
* (SORF) operation on an input 3d array that has been restructured for
* convolution on GPU. The input array should already live on GPU.
* The Hadamard transforms are performed using functions from basic_array_operations.cu
* the diagonal matrix multiplication is slightly
* different and so is implemented here.
*/
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#include "../basic_ops/basic_array_operations.h"
#include "convolution.h"

#define DEFAULT_THREADS_PER_BLOCK 256
#define DEFAULT_THREADS_PER_BLREDUCE 32




//Performs an elementwise multiplication of a row of a [3,1,P x S] array against the
//float [N,M,S] input array. Note that the dimensions must be checked before calling
//-- done by the wrapper -- and that only S elements of the appropriate row of
//the [3, 1, P x S] array are used.
template <typename T>
__global__ void conv1dMultiplyByRadem(T cArray[], int8_t *rademArray,
			int dim2, int startPosition, int numElements, float normConstant)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int8_t *rVal = rademArray + startPosition + (tid & (dim2 - 1));
    
    if (tid < numElements)
        cArray[tid] = cArray[tid] * *rVal * normConstant;
}


//This function performs the SORF block transform (HD3 HD2 HD1)
//on float input that has been reshaped as appropriate for a convolution.
//Note that reshapedX must have the same size across the
//last two dimensions as radem and its last dimension must
//be a power of two -- if those conditions are not met, you may
//get an unpredictable result! The Cython wrapper checks all
//of these criteria.
template <typename T>
const char *conv1dPrep(int8_t *radem, T reshapedX[], int reshapedDim0, 
                int reshapedDim1, int reshapedDim2, int startPosition,
                int numFreqs){

    int numElements = reshapedDim0 * reshapedDim1 * reshapedDim2;
    //This is the Hadamard normalization constant.
    float normConstant = log2(reshapedDim2) / 2;
    normConstant = 1 / pow(2, normConstant);
    int blocksPerGrid = (numElements + DEFAULT_THREADS_PER_BLOCK - 1) / 
                DEFAULT_THREADS_PER_BLOCK;
    
    //Multiply by first row of radem.
    conv1dMultiplyByRadem<T><<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(reshapedX, 
                        radem, reshapedDim2, startPosition, numElements,
                        normConstant);
    //First H-transform.
    cudaHTransform3d<T>(reshapedX, reshapedDim0, reshapedDim1, reshapedDim2);

    //Multiply by second row of radem.
    conv1dMultiplyByRadem<T><<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(reshapedX, 
                        radem + numFreqs, reshapedDim2, startPosition, numElements,
                        normConstant);
    //Second H-transform.
    cudaHTransform3d<T>(reshapedX, reshapedDim0, reshapedDim1, reshapedDim2);
        
    //Multiply by third row of radem.
    conv1dMultiplyByRadem<T><<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(reshapedX, 
                        radem + 2 * numFreqs, reshapedDim2, startPosition, numElements,
                        normConstant);
    //Last H-transform. Transform is in place so do not need to return anything except no error message.
    cudaHTransform3d<T>(reshapedX, reshapedDim0, reshapedDim1, reshapedDim2);
    return "no_error";
}
//Explicitly instantiate so wrapper can use.
template const char *conv1dPrep<float>(int8_t *radem, float reshapedX[], int reshapedDim0, 
                int reshapedDim1, int reshapedDim2, int startPosition,
                int numFreqs);
template const char *conv1dPrep<double>(int8_t *radem, double reshapedX[], int reshapedDim0, 
                int reshapedDim1, int reshapedDim2, int startPosition,
                int numFreqs);
