/*
* Contains functions needed to generate approximate polynomial kernel features on GPU.
*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "polynomial_operations.h"
#include "../basic_ops/basic_array_operations.h"

#define DEFAULT_THREADS_PER_BLOCK 256


//Performs an elementwise multiplication of a [c,M,P] array against the
//[N,M,P] input array or a [P] array against the [N,P] input array.
//Note that we mutiiply by the Hadamard normalization constant here.
template <typename T>
__global__ void polyMultByDiagRademMat(T cArray[], int8_t *rademArray,
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


//Performs an elementwise multiplication of a [c,M,P] array against the
//[N,M,P] input array or a [P] array against the [N,P] input array,
//WHILE copying from the input array into a copy buffer.
//Note that we mutiiply by the Hadamard normalization constant here.
template <typename T>
__global__ void polyMultAndCopyDiagRademMat(T cArray[], T copyBuffer[],
            int8_t *rademArray, int numElementsPerRow,
            int numElements, T normConstant)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int rVal, position;
    
    position = tid % numElementsPerRow;
    
    if (tid < numElements){
        rVal = rademArray[position];
        copyBuffer[tid] = cArray[tid] * rVal * normConstant;
    }
}



// Copies the copy buffer into the output array while multiplying
// by chiArr.
template <typename T>
__global__ void polyOutArrayCopyTransfer(T copyBuffer[],
            T chiArr[], double *outArray,
            int numFreqs, int numPerRow,
            int numElements)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int numRowsTraversed, numExcess;
    int cBuffPosition;
    
    numRowsTraversed = tid / numFreqs;
    numExcess = tid % numFreqs;
    cBuffPosition = numRowsTraversed * numPerRow + numExcess;

    if (tid < numElements)
        outArray[tid] = chiArr[numExcess] * copyBuffer[cBuffPosition];
}


// Multiplies the output array by copyBuffer and chiArr.
template <typename T>
__global__ void polyOutArrayMultTransfer(T copyBuffer[],
            T chiArr[], double *outArray,
            int numFreqs, int numPerRow,
            int numElements, int repeatNum)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int numRowsTraversed, numExcess, chiPosition;
    int cBuffPosition;
    
    numRowsTraversed = tid / numFreqs;
    numExcess = tid % numFreqs;
    cBuffPosition = numRowsTraversed * numPerRow + numExcess;
    chiPosition = repeatNum * numPerRow + numExcess;

    if (tid < numElements)
        outArray[tid] *= chiArr[chiPosition] * copyBuffer[cBuffPosition];
}







//Performs feature generation for an approximate polynomial kernel.
template <typename T>
const char *approxPolynomial_(int8_t *radem, T reshapedX[],
        T copyBuffer[], T chiArr[], double *outArray,
        int polydegree, int reshapedDim0, int reshapedDim1,
        int reshapedDim2, int numFreqs){
    int numElementsPerRow = reshapedDim1 * reshapedDim2;
    int numElements = reshapedDim0 * numElementsPerRow;
    int numOutputElements = numFreqs * reshapedDim0;

    //This is the Hadamard normalization constant.
    T normConstant = log2(reshapedDim2) / 2;
    normConstant = 1 / pow(2, normConstant);
    int blocksPerGrid = (numElements + DEFAULT_THREADS_PER_BLOCK - 1) / DEFAULT_THREADS_PER_BLOCK;
    int outputBlocks = (numOutputElements + DEFAULT_THREADS_PER_BLOCK - 1) / DEFAULT_THREADS_PER_BLOCK;
    //cudaProfilerStart();
    
    // First, copy the input into copy buffer and perform the initial SORF operation.
    // Then copy the results into output array while multiplying by chiArr.
    polyMultAndCopyDiagRademMat<T><<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(reshapedX,
                copyBuffer, radem, numElementsPerRow, numElements, normConstant);
    cudaHTransform3d<T>(copyBuffer, reshapedDim0, reshapedDim1, reshapedDim2);
    for (int k=1; k < 3; k++){
        polyMultByDiagRademMat<T><<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(copyBuffer,
                radem + k * numElementsPerRow,
                numElementsPerRow, numElements, normConstant);
        cudaHTransform3d<T>(copyBuffer, reshapedDim0, reshapedDim1, reshapedDim2);
    }
    
    polyOutArrayCopyTransfer<T><<<outputBlocks, DEFAULT_THREADS_PER_BLOCK>>>(copyBuffer,
            chiArr, outArray, numFreqs, numElementsPerRow,
            numElements);

    // Next, repeat this operation but multiplying the contents of outArray by
    // the results of the SORF operation on copyBuffer; do this up to polydegree
    // times.
    for (int i=1; i < polydegree; i++){
        polyMultAndCopyDiagRademMat<T><<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(reshapedX,
                copyBuffer, radem + (i * 3 * numElementsPerRow),
                numElementsPerRow, numElements, normConstant);
        cudaHTransform3d<T>(copyBuffer, reshapedDim0, reshapedDim1, reshapedDim2);
        for (int k=1; k < 3; k++){
            polyMultByDiagRademMat<T><<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(copyBuffer,
                radem + (i * 3 + k) * numElementsPerRow,
                numElementsPerRow, numElements, normConstant);
            cudaHTransform3d<T>(copyBuffer, reshapedDim0, reshapedDim1, reshapedDim2);
        }
        polyOutArrayMultTransfer<T><<<outputBlocks, DEFAULT_THREADS_PER_BLOCK>>>(copyBuffer,
            chiArr, outArray, numFreqs, numElementsPerRow,
            numElements, i);
    }

    //cudaProfilerStop();
    return "no_error";
}
//Instantiate templates explicitly so wrapper can use.
template const char *approxPolynomial_<float>(int8_t *radem, float reshapedX[],
        float copyBuffer[], float chiArr[], double *outArray,
        int polydegree, int reshapedDim0, int reshapedDim1,
        int reshapedDim2, int numFreqs);
template const char *approxPolynomial_<double>(int8_t *radem, double reshapedX[],
        double copyBuffer[], double chiArr[], double *outArray,
        int polydegree, int reshapedDim0, int reshapedDim1,
        int reshapedDim2, int numFreqs);
