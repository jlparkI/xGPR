/*
* Contains tools for basic array operations on GPU -- fast hadamard transforms
* and SRHT. Note that many operations here assume specific dimensions
* of the input array are a power of 2. The Cython wrapper checks this, so do
* not call these routines OUTSIDE of the Cython wrapper -- use the Cython wrapper.
*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <math.h>
#include "../shared_constants.h"
#include "basic_array_operations.h"
#include "sharedmem.h"



//Uses shared memory to perform a reasonably efficient single kernel
//transform up to MAX_BASE_LEVEL_TRANSFORM, then uses a somewhat
//less efficient global memory procedure to complete the
//transform if needed. Performs the transform on a single
//vector of the input.
template <typename T>
__global__ void hadamardTransform(T cArray[], int N, int log2N){
    int stepSize = MIN(N, MAX_BASE_LEVEL_TRANSFORM);
    int nRepeats = N / stepSize;

    SharedMemory<T> shared;
    T *s_data = shared.getPointer();
    int spacing, pos = threadIdx.x;
    int lo, id1, id2;
    T *src_ptr = cArray + (blockIdx.x << log2N);
    T y;

    for (int rep = 0; rep < nRepeats; rep++){
        for (int i = threadIdx.x; i < stepSize; i += blockDim.x)
            s_data[i] = src_ptr[i];

        id1 = (pos << 1);
        id2 = id1 + 1;
        __syncthreads();
        y = s_data[id2];
        s_data[id2] = s_data[id1] - y;
        s_data[id1] += y;


        for (spacing = 2; spacing < stepSize; spacing <<= 1){
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
        __syncthreads();
        for (int i = threadIdx.x; i < stepSize; i += blockDim.x)
            src_ptr[i] = s_data[i];
        __syncthreads();
        src_ptr += stepSize;
    }

    if (N > MAX_BASE_LEVEL_TRANSFORM){
        src_ptr = cArray + (blockIdx.x << log2N);

        for (int spacing = stepSize; spacing < N; spacing <<= 1){
            __syncthreads();

            for (int k = 0; k < N; k += (spacing << 1)){
                for (int i = threadIdx.x; i < spacing; i += blockDim.x){
                    id1 = i+k;
                    id2 = id1 + spacing;
                    y = src_ptr[id2];
                    src_ptr[id2] = src_ptr[id1] - y;
                    src_ptr[id1] += y;
                }
            }
        }
    }
}


//Performs a Hadamard transform on a 2d array after first multiplying
//by a diagonal array with entries populated from a Rademacher distribution.
//This is intended for use with SRHT and related procedures.
template <typename T>
__global__ void hadamardTransformRadMult(T cArray[], int N, int log2N,
        const int8_t *radem, T normConstant){
    int stepSize = MIN(N, MAX_BASE_LEVEL_TRANSFORM);
    int nRepeats = N / stepSize;

    SharedMemory<T> shared;
    T *s_data = shared.getPointer();
    int spacing, pos = threadIdx.x;
    int lo, id1, id2;
    T *src_ptr = cArray + (blockIdx.x << log2N);
    T y;

    const int8_t *rademPtr = radem;

    for (int rep = 0; rep < nRepeats; rep++){
        for (int i = threadIdx.x; i < stepSize; i += blockDim.x)
            s_data[i] = src_ptr[i];
    
        //Multiply by the diagonal array here.
        for (int i = threadIdx.x; i < stepSize; i += blockDim.x)
            s_data[i] = s_data[i] * rademPtr[i] * normConstant;

        rademPtr += stepSize;

        id1 = (pos << 1);
        id2 = id1 + 1;
        __syncthreads();
        y = s_data[id2];
        s_data[id2] = s_data[id1] - y;
        s_data[id1] += y;


        for (spacing = 2; spacing < stepSize; spacing <<= 1){
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
        __syncthreads();
        for (int i = threadIdx.x; i < stepSize; i += blockDim.x)
            src_ptr[i] = s_data[i];
        __syncthreads();
        src_ptr += stepSize;
    }

    if (N > MAX_BASE_LEVEL_TRANSFORM){
        src_ptr = cArray + (blockIdx.x << log2N);

        for (int spacing = stepSize; spacing < N; spacing <<= 1){
            __syncthreads();

            for (int k = 0; k < N; k += (spacing << 1)){
                for (int i = threadIdx.x; i < spacing; i += blockDim.x){
                    id1 = i+k;
                    id2 = id1 + spacing;
                    y = src_ptr[id2];
                    src_ptr[id2] = src_ptr[id1] - y;
                    src_ptr[id1] += y;
                }
            }
        }
    }
}


//Performs an elementwise multiplication of a [c,M,P] array against the
//[N,M,P] input array or a [P] array against the [N,P] input array.
//Note that the last dimensions of these must be the
//same, and this function does not check this -- caller must check. Note that
//we multiply by the Hadamard normalization constant here.
template <typename T>
__global__ void diagonalRademMultiply(T cArray[], const int8_t *rademArray,
			int numElementsPerRow, int numElements, T normConstant)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int position = tid % numElementsPerRow;

    if (tid < numElements)
        cArray[tid] = cArray[tid] * rademArray[position] * normConstant;
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
//of cArray which must be 2d; we expect cArray.shape[1] to 
//be a power of 2 (caller must verify).
template <typename T>
void cudaHTransform(T cArray[],
		int dim0, int dim1, int dim2){

    int stepSize, log2N;
    stepSize = MIN(MAX_BASE_LEVEL_TRANSFORM, dim1);
    log2N = log2(dim1);

    hadamardTransform<T><<<dim0, stepSize / 2,
                    stepSize * sizeof(T)>>>(cArray, dim1, log2N);
}
//Instantiate templates explicitly so wrapper can use.
template void cudaHTransform<float>(float cArray[],
                int dim0, int dim1, int dim2);
template void cudaHTransform<double>(double cArray[],
                int dim0, int dim1, int dim2);



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
const char *cudaSRHT2d(T cArray[], const int8_t *radem,
                int dim0, int dim1){
    //This is the Hadamard normalization constant.
    T normConstant = log2(dim1) / 2;
    normConstant = 1 / pow(2, normConstant);
    int stepSize, log2N;
    stepSize = MIN(MAX_BASE_LEVEL_TRANSFORM, dim1);
    log2N = log2(dim1);


    //cudaProfilerStart();
    hadamardTransformRadMult<T><<<dim0, stepSize / 2,
        stepSize * sizeof(T)>>>(cArray, dim1, log2N,
                    radem, normConstant);


    //cudaProfilerStop();
    return "no_error";
}
//Instantiate templates explicitly so wrapper can use.
template const char *cudaSRHT2d<float>(float cArray[], const int8_t *radem,
                int dim0, int dim1);
template const char *cudaSRHT2d<double>(double cArray[], const int8_t *radem,
                int dim0, int dim1);



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
const char *cudaSORF3d(T cArray[], const int8_t *radem,
                int dim0, int dim1, int dim2){
    int numElementsPerRow = dim1 * dim2;
    int numElements = dim1 * dim2 * dim0;
    //This is the Hadamard normalization constant.
    T normConstant = log2(dim2) / 2;
    normConstant = 1 / pow(2, normConstant);
    int blocksPerGrid = (numElements + DEFAULT_THREADS_PER_BLOCK - 1) / DEFAULT_THREADS_PER_BLOCK;
    //cudaProfilerStart();

    //else{
        //cudaHTransformWithDiagMultiply(cArray, dim0, dim1, dim2,
        //        radem, normConstant, numElementsPerRow);
        //cudaHTransformWithDiagMultiply(cArray, dim0, dim1, dim2,
        //        radem + numElementsPerRow, normConstant,
        //        numElementsPerRow);
        //cudaHTransformWithDiagMultiply(cArray, dim0, dim1, dim2,
        //        radem + 2 * numElementsPerRow, normConstant,
        //        numElementsPerRow);
    //}

    //cudaProfilerStop();
    return "no_error";
}
//Instantiate templates explicitly so wrapper can use.
template const char *cudaSORF3d<float>(float cArray[], const int8_t *radem,
                int dim0, int dim1, int dim2);
template const char *cudaSORF3d<double>(double cArray[], const int8_t *radem,
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
const char *cudaConvSORF3d(T cArray[], const int8_t *radem,
                int dim0, int dim1, int dim2,
                int startPosition, int numElements,
                int rademShape2, T normConstant){
    int blocksPerGrid = (numElements + DEFAULT_THREADS_PER_BLOCK - 1) / DEFAULT_THREADS_PER_BLOCK;
    //cudaProfilerStart();

    if (dim2 <= MAX_SINGLE_STAGE_TRANSFORM){
        int log2N = log2(dim2);
        blocksPerGrid = numElements / dim2;
        //singleStepConvSORF<T><<<blocksPerGrid, dim2 / 2, dim2 * sizeof(T)>>>(cArray, dim2, log2N,
        //        radem, normConstant, startPosition, rademShape2);
    }
    else{
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
    }
    //cudaProfilerStop();
    return "no_error";
}
//Instantiate templates explicitly so wrapper can use.
template const char *cudaConvSORF3d<float>(float cArray[], const int8_t *radem,
                int dim0, int dim1, int dim2, int startPosition,
                int numElements, int rademShape2, float normConstant);
template const char *cudaConvSORF3d<double>(double cArray[], const int8_t *radem,
                int dim0, int dim1, int dim2, int startPosition,
                int numElements, int rademShape2, double normConstant);
