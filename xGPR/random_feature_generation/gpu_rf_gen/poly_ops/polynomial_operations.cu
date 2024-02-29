/*
* Contains functions needed to generate approximate polynomial kernel features on GPU.
*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "polynomial_operations.h"
#include "../shared_constants.h"
#include "../basic_ops/basic_array_operations.h"




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
    int outputBlocks = (numOutputElements + DEFAULT_THREADS_PER_BLOCK - 1) / DEFAULT_THREADS_PER_BLOCK;
    const char *errCode;
    //cudaProfilerStart();
    
    // First, copy the input into copy buffer and perform the initial SORF operation.
    // Then copy the results into output array while multiplying by chiArr.
    cudaMemcpy(copyBuffer, reshapedX, sizeof(T) * numElements,
                cudaMemcpyDeviceToDevice);
    errCode = cudaSORF3d<T>(copyBuffer, radem,
            reshapedDim0, reshapedDim1, reshapedDim2);
    polyOutArrayCopyTransfer<T><<<outputBlocks, DEFAULT_THREADS_PER_BLOCK>>>(copyBuffer,
            chiArr, outArray, numFreqs, numElementsPerRow,
            numElements);

    // Next, repeat this operation but multiplying the contents of outArray by
    // the results of the SORF operation on copyBuffer; do this up to polydegree
    // times.
    for (int i=1; i < polydegree; i++){
        cudaMemcpy(copyBuffer, reshapedX, sizeof(T) * numElements,
                cudaMemcpyDeviceToDevice);
        errCode = cudaSORF3d<T>(copyBuffer, radem + (i * 3 * numElementsPerRow),
                reshapedDim0, reshapedDim1, reshapedDim2);
        polyOutArrayMultTransfer<T><<<outputBlocks, DEFAULT_THREADS_PER_BLOCK>>>(copyBuffer,
            chiArr, outArray, numFreqs, numElementsPerRow,
            numElements, i);
    }

    //cudaProfilerStop();
    return errCode;
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
