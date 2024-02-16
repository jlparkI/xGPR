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
    
    const char *errCode = cudaConvSORF3d<T>(reshapedX, radem,
                reshapedDim0, reshapedDim1, reshapedDim2,
                startPosition, numElements, numFreqs, normConstant);
    return errCode;
}
//Explicitly instantiate so wrapper can use.
template const char *conv1dPrep<float>(int8_t *radem, float reshapedX[], int reshapedDim0, 
                int reshapedDim1, int reshapedDim2, int startPosition,
                int numFreqs);
template const char *conv1dPrep<double>(int8_t *radem, double reshapedX[], int reshapedDim0, 
                int reshapedDim1, int reshapedDim2, int startPosition,
                int numFreqs);
