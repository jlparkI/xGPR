/*
* Contains all functions needed to run the structured orthogonal features
* (SORF) and randomized Hadamard transform (RHT) operations on an input 3d
* array on GPU. Note that many operations here assume specific dimensions
* of the input array are a power of 2. The Cython wrapper checks this, so do
* not call these routines OUTSIDE of the Cython wrapper -- use the Cython wrapper.
*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include "basic_array_operations.h"
#include "sharedmem.h"

#define DEFAULT_THREADS_PER_BLOCK 256
#define MAX_BASE_LEVEL_TRANSFORM 512



//Uses shared memory to perform a reasonably efficient single kernel
//transform covering strides up to MAX_BASE_LEVEL_TRANSFORM.
template <typename T>
__global__ void baseLevelTransform(T cArray[], int N, int log2N){
    int startPos = blockIdx.x << log2N;

    SharedMemory<T> shared;
    T *s_data = shared.getPointer();
    int i, spacing, pos = threadIdx.x;
    int lo, id1, id2;
    T *src_ptr = cArray + startPos;
    T y;

    for (i = threadIdx.x; i < N; i += blockDim.x)
        s_data[i] = src_ptr[i];

    id1 = (pos << 1);
    id2 = id1 + 1;
    __syncthreads();
    y = s_data[id2];
    s_data[id2] = s_data[id1] - y;
    s_data[id1] += y;


    for (spacing = 2; spacing < N; spacing <<= 1){
        //Equivalent to pos mod spacing if spacing is a power of 2,
        //which here is always true.
        lo = pos & (spacing - 1);
        id1 = ((pos - lo) << 1) + lo;
        id2 = id1 + spacing;
        __syncthreads();
        y = s_data[id2];
        s_data[id2] = s_data[id1] - y;
        s_data[id1] += y;
    }
    for (i = threadIdx.x; i < N; i += blockDim.x){
        src_ptr[i] = s_data[i];
    }
}



//Performs subsequent stages of the transform (strides
//> MAX_BASE_LEVEL_TRANSFORM) using a less efficient global
//memory procedure.
template <typename T>
__global__ void levelNTransform(T cArray[], int arrsize,
                                int spacing)
{
    int pos = blockDim.x * blockIdx.x + threadIdx.x;
    //Equivalent to pos mod spacing if spacing is a power of 2,
    //which here is always true.
    int lo = (pos & (spacing - 1));
    int id = lo + ((pos - lo) << 1);
    
    if (id < arrsize){
        T y, *cPtr = cArray + id;

        y = cPtr[spacing];
        cPtr[spacing] = *cPtr - y;
        *cPtr += y;
    }
}


//Performs an elementwise multiplication of a [c,M,P] array against the
//[N,M,P] input array or a [P] array against the [N,P] input array.
//Note that the last dimensions of these must be the
//same, and this function does not check this -- caller must check. Note that
//we mutiiply by the Hadamard normalization constant here.
template <typename T>
__global__ void diagonalRademMultiply(T cArray[], const int8_t *rademArray,
			int numElementsPerRow, int numElements, T normConstant)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int rVal, position;
    
    position = tid % numElementsPerRow;

    if (tid < numElements){
        rVal = rademArray[position];
        cArray[tid] = cArray[tid] * rVal * normConstant;
    }
}


//Performs an elementwise multiplication of a row of a [3,1,P x S] array against the
//float [N,M,S] input array. Note that the dimensions must be checked before calling
//-- done by the wrapper -- and that only S elements of the appropriate row of
//the [3, 1, P x S] array are used.
template <typename T>
__global__ void conv1dDiagonalRademMultiply(T cArray[],
            const int8_t *rademArray,
			int dim2, int startPosition, int numElements,
            T normConstant)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int position = startPosition + (tid & (dim2 - 1));
    
    if (tid < numElements)
        cArray[tid] = cArray[tid] * rademArray[position] * normConstant;
}




//We perform the transform over the last dimension
//of cArray which must be 3d; we expect cArray.shape[2] to 
//be a power of 2 (caller must verify). This can also be used
//for 2d arrays by passing dim1=1.
template <typename T>
void cudaHTransform(T cArray[],
		int dim0, int dim1, int dim2){

    int N, log2N;
    int spacing = 1;
    int arrsize = dim0 * dim1 * dim2;
    int blocksPerGrid;


    //baseLevelTransform only covers strides up to MAX_BASE_LEVEL_TRANSFORM.
    //If dim2 is less than that, we're set. Otherwise, run baseLevelTransform
    //first then use a somewhat slower global memory procedure for larger strides.
    N = (MAX_BASE_LEVEL_TRANSFORM < dim2) ? MAX_BASE_LEVEL_TRANSFORM : dim2;
    log2N = log2(N);
    blocksPerGrid = arrsize / N;

    baseLevelTransform<T><<<blocksPerGrid, N / 2, 
                    N * sizeof(T)>>>(cArray, N, log2N);
    
    if (dim2 <= MAX_BASE_LEVEL_TRANSFORM)
        return;
    
    //The largest strides (for large dim2) are handled by a somewhat
    //slower global memory procedure.
    spacing = MAX_BASE_LEVEL_TRANSFORM;
    blocksPerGrid = (arrsize / 2) + DEFAULT_THREADS_PER_BLOCK - 1;
    blocksPerGrid /= DEFAULT_THREADS_PER_BLOCK;
    while (spacing < dim2){
        levelNTransform<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(cArray, 
                                arrsize, spacing);
        spacing <<= 1;
    }
}



//This function performs the SORF block transform (HD3 HD2 HD1) 
//Note that cArray must have the same size across the
//last two dimensions as radem and its last dimension must
//be a power of two -- if those conditions are not met, you may
//get an unpredictable result! The Cython wrapper checks all
//of these criteria -- any other caller using this function
//should do the same.
//
//Note that all of these arrays are already expected to "live" on GPU.
//Can be used to perform SORF on a 2d array by passing dim1=1.
template <typename T>
const char *cudaSORF3d(T cArray[], int8_t *radem,
                int dim0, int dim1, int dim2){
    int numElementsPerRow = dim1 * dim2;
    int numElements = dim1 * dim2 * dim0;
    //This is the Hadamard normalization constant.
    T normConstant = log2(dim2) / 2;
    normConstant = 1 / pow(2, normConstant);
    int blocksPerGrid = (numElements + DEFAULT_THREADS_PER_BLOCK - 1) / DEFAULT_THREADS_PER_BLOCK;
    //cudaProfilerStart();

    //Multiply by D1.
    diagonalRademMultiply<T><<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(cArray, radem, 
                                 numElementsPerRow, numElements, normConstant);
    
    //First H-transform.
    cudaHTransform<T>(cArray, dim0, dim1, dim2);
    
    //Multiply by D2.
    diagonalRademMultiply<T><<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(cArray, radem + numElementsPerRow,
                                 numElementsPerRow, numElements, normConstant);

    //Second H-transform.
    cudaHTransform<T>(cArray, dim0, dim1, dim2);
    
    //Multiply by D3.
    diagonalRademMultiply<T><<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(cArray, radem + 2 * numElementsPerRow,
                                 numElementsPerRow, numElements, normConstant);
    
    //Last H-transform. Transform is in place so do not need to return anything except no error message.
    cudaHTransform<T>(cArray, dim0, dim1, dim2); 

    //cudaProfilerStop();
    return "no_error";
}
//Instantiate templates explicitly so wrapper can use.
template const char *cudaSORF3d<float>(float cArray[], int8_t *radem,
                int dim0, int dim1, int dim2);
template const char *cudaSORF3d<double>(double cArray[], int8_t *radem,
                int dim0, int dim1, int dim2);



//This function performs the SORF block transform (HD3 HD2 HD1)
//but for convolution-type operations.
//Note that cArray must have the same size across the
//last dimension as radem and its last dimension must
//be a power of two -- if those conditions are not met, you may
//get an unpredictable result! The Cython wrapper checks all
//of these criteria -- any other caller using this function
//should do the same.
//
//Note that all of these arrays are already expected to "live" on GPU.
template <typename T>
const char *cudaConvSORF3d(T cArray[], int8_t *radem,
                int dim0, int dim1, int dim2,
                int startPosition, int numElements,
                int rademShape2, T normConstant){
    int blocksPerGrid = (numElements + DEFAULT_THREADS_PER_BLOCK - 1) / DEFAULT_THREADS_PER_BLOCK;
    //cudaProfilerStart();
        
    //Multiply by first row of radem.
    conv1dDiagonalRademMultiply<T><<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(cArray, 
                        radem, dim2, startPosition, numElements,
                        normConstant);

    //First H-transform.
    cudaHTransform<T>(cArray, dim0, dim1, dim2);

    //Multiply by second row of radem.
    conv1dDiagonalRademMultiply<T><<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(cArray, 
                        radem + rademShape2, dim2, startPosition, numElements,
                        normConstant);
    //Second H-transform.
    cudaHTransform<T>(cArray, dim0, dim1, dim2);
        
    //Multiply by third row of radem.
    conv1dDiagonalRademMultiply<T><<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(cArray,
                        radem + 2 * rademShape2, dim2, startPosition, numElements,
                        normConstant);
    //Last H-transform.
    cudaHTransform<T>(cArray, dim0, dim1, dim2);

    //cudaProfilerStop();
    return "no_error";
}
//Instantiate templates explicitly so wrapper can use.
template const char *cudaConvSORF3d<float>(float cArray[], int8_t *radem,
                int dim0, int dim1, int dim2, int startPosition,
                int numElements, int rademShape2, float normConstant);
template const char *cudaConvSORF3d<double>(double cArray[], int8_t *radem,
                int dim0, int dim1, int dim2, int startPosition,
                int numElements, int rademShape2, double normConstant);





//Performs the first two steps of SRHT (HD)
//Note that cArray must have the same size across the
//last two dimensions as radem and its last dimension must
//be a power of two -- if those conditions are not met, you may
//get an unpredictable result! The Cython wrapper checks all
//of these criteria -- any other caller using this function
//should do the same.
//
//Note that all of these arrays are already expected to "live" on GPU.
template <typename T>
const char *cudaSRHT2d(T cArray[], int8_t *radem,
                int dim0, int dim1){
    int numElementsPerRow = dim1;
    int numElements = dim1 * dim0;
    //This is the Hadamard normalization constant.
    T normConstant = log2(dim1) / 2;
    normConstant = 1 / pow(2, normConstant);
    int blocksPerGrid = (numElements + DEFAULT_THREADS_PER_BLOCK - 1) / DEFAULT_THREADS_PER_BLOCK;

    //cudaProfilerStart();

    //Multiply by D1.
    diagonalRademMultiply<T><<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(cArray, radem, 
                                 numElementsPerRow, numElements, normConstant);
    
    //First H-transform.
    cudaHTransform<T>(cArray, dim0, 1, dim1);

    //cudaProfilerStop();
    return "no_error";
}
//Instantiate templates explicitly so wrapper can use.
template const char *cudaSRHT2d<float>(float cArray[], int8_t *radem,
                int dim0, int dim1);
template const char *cudaSRHT2d<double>(double cArray[], int8_t *radem,
                int dim0, int dim1);
