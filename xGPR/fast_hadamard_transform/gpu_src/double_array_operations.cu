/*
* Contains all functions needed to run the structured orthogonal features
* (SORF) and randomized Hadamard transform (RHT) operations on an input 3d
* array of doubles on GPU. The input array should
* already live on GPU.
*/

//Note that where possible, we have
//avoided use of modulo and integer division
//because they are more expensive. Instead, we use
//(location >> log2Spacing) (equivalent to floor division on
//location / spacing IF spacing is a power of 2) and 
//(location & (spacing - 1)) 
//(equivalent to location % spacing IF spacing is a power of 2). 
//Note that all of this like many operations here works ONLY
//if array.shape[2] is a power of 2 -- this should
//ALWAYS be checked by caller. The Cython wrapper checks this
//(and many other crucial details).
//If you decide to use this outside of the Cython wrapper,
//you must check yourself.

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "double_array_operations.h"
#include <cuda_profiler_api.h>

#define DEFAULT_THREADS_PER_BLOCK 256
#define MAX_BASE_LEVEL_TRANSFORM 512



//Uses shared memory to perform a reasonably efficient single kernel
//transform covering strides up to MAX_BASE_LEVEL_TRANSFORM.
__global__ void doubleBaseLevelTransform(double *cArray, int N, int log2N){
    int startPos = blockIdx.x << log2N;

    extern __shared__ double s_data[];
    int i, spacing, pos = threadIdx.x;
    int lo, id1, id2;
    double *src_ptr = cArray + startPos;
    double y;

    for (i = threadIdx.x; i < N; i += blockDim.x)
        s_data[i] = src_ptr[i];

    id1 = (pos << 1);
    id2 = id1 + 1;
    __syncthreads();
    y = s_data[id2];
    s_data[id2] = s_data[id1] - y;
    s_data[id1] += y;


    for (spacing = 2; spacing < N; spacing <<= 1){
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

//Combines the level 2 transform and the level 4 transform in
//global memory for arrays with small sizes where shape[2] < 32
//so that the shared memory procedure is not efficient.
__global__ void doubleShape4Transform(double *cArray, int arrsize)
{
    int id = 4 * (blockDim.x * blockIdx.x + threadIdx.x);
    int id1 = id, id2 = id + 1;
    double y;
    if (id < arrsize){
        y = cArray[id2];
        cArray[id2] = cArray[id1] - y;
        cArray[id1] += y;
        id1 += 2;
        id2 += 2;

        y = cArray[id2];
        cArray[id2] = cArray[id1] - y;
        cArray[id1] += y;
        id1 -= 1;

        y = cArray[id2];
        cArray[id2] = cArray[id1] - y;
        cArray[id1] += y;
        id1 -= 1;
        id2 -= 1;

        y = cArray[id2];
        cArray[id2] = cArray[id1] - y;
        cArray[id1] += y;
    }
}



//Performs subsequent stages of the transform (strides
//> MAX_BASE_LEVEL_TRANSFORM) using a less efficient global
//memory procedure.
__global__ void doubleLevelNTransform(double *cArray, int arrsize,
                                int spacing)
{
    int pos = blockDim.x * blockIdx.x + threadIdx.x;
    int lo = (pos & (spacing - 1));
    int id = lo + ((pos - lo) << 1);
    double y, *cPtr = cArray + id;
    
    if (id < arrsize){
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
__global__ void doubleMultiplyByDiagonalRademacherMat(double *cArray, int8_t *rademArray,
			int numElementsPerRow, int numElements, double normConstant)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int rVal, position;
    
    position = j % numElementsPerRow;
    rVal = rademArray[position];
    if (j < numElements)
        cArray[j] = cArray[j] * rVal * normConstant;
}


//Performs an elementwise multiplication of a [c,M,P] array against the
//[N,M,P] input array or a [P] array against the [N,P] input array.
//Note that the last dimensions of these must be the
//same, and this function does not check this -- caller must check. Note that
//we multiply by the Hadamard normalization constant here.
__global__ void doubleMultiplyByDiagonalMat(double *cArray, double *diagArray,
			int numElementsPerRow, int numElements, double normConstant)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    double rVal;
    int position;
    
    position = j % numElementsPerRow;
    rVal = diagArray[position];
    if (j < numElements)
        cArray[j] = cArray[j] * rVal * normConstant;
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
const char *doubleCudaSORF3d(double *cArray, int8_t *radem,
                int dim0, int dim1, int dim2){
    int numElementsPerRow = dim1 * dim2;
    int numElements = dim1 * dim2 * dim0;
    //This is the Hadamard normalization constant.
    double normConstant = log2(dim2) / 2;
    normConstant = 1 / pow(2, normConstant);
    int blocksPerGrid = (numElements + DEFAULT_THREADS_PER_BLOCK - 1) / DEFAULT_THREADS_PER_BLOCK;
    //cudaProfilerStart();

    //Multiply by D1.
    doubleMultiplyByDiagonalRademacherMat<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(cArray, radem, 
                                 numElementsPerRow, numElements, normConstant);
    
    //First H-transform.
    doubleCudaHTransform3d(cArray, dim0, dim1, dim2);
    
    //Multiply by D2.
    doubleMultiplyByDiagonalRademacherMat<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(cArray, radem + numElementsPerRow,
                                 numElementsPerRow, numElements, normConstant);

    //Second H-transform.
    doubleCudaHTransform3d(cArray, dim0, dim1, dim2);
    
    //Multiply by D3.
    doubleMultiplyByDiagonalRademacherMat<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(cArray, radem + 2 * numElementsPerRow,
                                 numElementsPerRow, numElements, normConstant);
    
    //Last H-transform. Transform is in place so do not need to return anything except no error message.
    doubleCudaHTransform3d(cArray, dim0, dim1, dim2); 

    //cudaProfilerStop();
    return "no_error";
}


//Performs the first two steps of SRHT (HD)
//Note that cArray must have the same size across the
//last two dimensions as radem and its last dimension must
//be a power of two -- if those conditions are not met, you may
//get an unpredictable result! The Cython wrapper checks all
//of these criteria -- any other caller using this function
//should do the same.
//
//Note that all of these arrays are already expected to "live" on GPU.
const char *doubleCudaSRHT2d(double *cArray, int8_t *radem,
                int dim0, int dim1){
    int numElementsPerRow = dim1;
    int numElements = dim1 * dim0;
    //This is the Hadamard normalization constant.
    double normConstant = log2(dim1) / 2;
    normConstant = 1 / pow(2, normConstant);
    int blocksPerGrid = (numElements + DEFAULT_THREADS_PER_BLOCK - 1) / DEFAULT_THREADS_PER_BLOCK;
    //cudaProfilerStart();

    //Multiply by D1.
    doubleMultiplyByDiagonalRademacherMat<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(cArray, radem, 
                                 numElementsPerRow, numElements, normConstant);
    
    //First H-transform.
    doubleCudaHTransform2d(cArray, dim0, dim1);

    //cudaProfilerStop();
    return "no_error";
}



//We perform the transform over the last dimension
//of cArray which must be 3d; we expect cArray.shape[2] to 
//be a power of 2 (caller must verify).
void doubleCudaHTransform3d(double *cArray,
		int dim0, int dim1, int dim2){

    int N, log2N;
    int spacing = 1;
    int arrsize = dim0 * dim1 * dim2;
    int blocksPerGrid;

    //For less than 64, use specialized routines. dim2 is always
    //a power of two, and for best performance on CUDA threads per block
    //should be a multiple of 32, so the baseLevelTransform does
    //not work as well for dim2 < 64. There is a great deal of room
    //for additional optimization here that we have not done (yet) 
    //because input dim < 32 but > 2 is a somewhat niche application.
    if (dim2 < 32){
        blocksPerGrid = getNumBlocksDoubleTransform(arrsize, 2);
        if (dim2 == 2){
            doubleLevelNTransform<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(cArray, 
                                arrsize, 1);
        }
        else{
            doubleShape4Transform<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(cArray, 
                                arrsize);
            spacing = 4;
            while (spacing < dim2){
                doubleLevelNTransform<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(cArray, 
                                arrsize, spacing);
                spacing <<= 1;
            }
        }
        return;
    }

    //Otherwise, we use the baseLevelTransform, which uses shared
    //memory and is relatively efficient. baseLevelTransform only
    //covers strides up to MAX_BASE_LEVEL_TRANSFORM. If dim2 is less than that,
    //we're set. Otherwise, run baseLevelTransform first then use
    //a somewhat slower global memory procedure for larger strides.
    N = (MAX_BASE_LEVEL_TRANSFORM < dim2) ? MAX_BASE_LEVEL_TRANSFORM : dim2;
    log2N = log2(N);
    blocksPerGrid = arrsize / N;

    doubleBaseLevelTransform<<<blocksPerGrid, N / 2, 
                    N * sizeof(double)>>>(cArray, N, log2N);
    
    if (dim2 <= MAX_BASE_LEVEL_TRANSFORM)
        return;
    
    //The largest strides (for large dim2) are handled by a somewhat
    //slower global memory procedure.
    spacing = MAX_BASE_LEVEL_TRANSFORM;
    blocksPerGrid = getNumBlocksDoubleTransform(arrsize, 2);
    while (spacing < dim2){
        doubleLevelNTransform<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(cArray, 
                                arrsize, spacing);
        spacing <<= 1;
    }
}


//We perform the transform over the last dimension
//of cArray which must be 2d; we expect cArray.shape[1] to 
//be a power of 2 (caller must verify).
void doubleCudaHTransform2d(double *cArray, int dim0, int dim1){

    int N, log2N;
    int spacing = 1;
    int arrsize = dim0 * dim1;
    int blocksPerGrid;

    //For less than 64, use specialized routines. dim2 is always
    //a power of two, and for best performance on CUDA threads per block
    //should be a multiple of 32, so the baseLevelTransform does
    //not work as well for dim2 < 64. There is a great deal of room
    //for additional optimization here that we have not done (yet) 
    //because input dim < 32 but > 2 is a somewhat niche application.
    if (dim1 < 32){
        blocksPerGrid = getNumBlocksDoubleTransform(arrsize, 2);
        if (dim1 == 2){
            doubleLevelNTransform<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(cArray, 
                                arrsize, 1);
        }
        else{
            doubleShape4Transform<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(cArray, 
                                arrsize);
            spacing = 4;
            while (spacing < dim1){
                doubleLevelNTransform<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(cArray, 
                                arrsize, spacing);
                spacing <<= 1;
            }
        }
        return;
    }

    //Otherwise, we use the baseLevelTransform, which uses shared
    //memory and is relatively efficient. baseLevelTransform only
    //covers strides up to MAX_BASE_LEVEL_TRANSFORM. If dim2 is less than that,
    //we're set. Otherwise, run baseLevelTransform first then use
    //a somewhat slower global memory procedure for larger strides.
    N = (MAX_BASE_LEVEL_TRANSFORM < dim1) ? MAX_BASE_LEVEL_TRANSFORM : dim1;
    log2N = log2(N);
    blocksPerGrid = arrsize / N;

    doubleBaseLevelTransform<<<blocksPerGrid, N / 2, 
                    N * sizeof(double)>>>(cArray, N, log2N);
    
    if (dim1 <= MAX_BASE_LEVEL_TRANSFORM)
        return;
    
    //The largest strides (for large dim1) are handled by a somewhat
    //slower global memory procedure.
    spacing = MAX_BASE_LEVEL_TRANSFORM;
    blocksPerGrid = getNumBlocksDoubleTransform(arrsize, 2);
    while (spacing < dim1){
        doubleLevelNTransform<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(cArray, 
                                arrsize, spacing);
        spacing <<= 1;
    }
}





//Calculates the number of blocks for all transforms except the 
//baseLevelTransform, which uses shared memory and hence
//a slightly different procedure.
int getNumBlocksDoubleTransform(int arrsize, int divisor){

    int blocksPerGrid;
    blocksPerGrid = (arrsize / divisor) + DEFAULT_THREADS_PER_BLOCK - 1;
    return blocksPerGrid / DEFAULT_THREADS_PER_BLOCK;
}
