/*
* Contains functions needed to generate exact quadratic polynomial
* features and approximate polynomial kernel features on GPU.
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
    rVal = rademArray[position];
    if (tid < numElements)
        cArray[tid] = cArray[tid] * rVal * normConstant;
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
    rVal = rademArray[position];
    if (tid < numElements)
        copyBuffer[tid] = cArray[tid] * rVal * normConstant;
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



//Generates the features for the exact quadratic.
template <typename T>
__global__ void genExactQuadFeatures(T inArray[], double *outArray,
        int inDim1, int outDim1, int numElements){
    int pos = blockDim.x * blockIdx.x + threadIdx.x;
    int rowNum = pos / inDim1;
    int positionInRow = pos % inDim1;
    T inVal1 = inArray[pos];
    T *inPtr = inArray + pos;
    double *outPtr = outArray + rowNum * outDim1;
    for (int i=0; i < positionInRow; i++)
        outPtr += inDim1 + 1 - i;
    
    if (pos < numElements){
        *outPtr = inVal1;
        outPtr++;
        for (int i=positionInRow; i < inDim1; i++){
            *outPtr = inVal1 * *inPtr;
            outPtr++;
            inPtr++;
        }
    }
}



//Performs feature generation for an exact quadratic (i.e. polynomial
//regression that is exact, not approximated).
//
//Note that all of these arrays are already expected to "live" on GPU.
template <typename T>
const char *cudaExactQuadratic_(T inArray[], double *outArray,
                    int inDim0, int inDim1){
    int numInteractions = (inDim1 * (inDim1 - 1)) / 2;
    int outDim1 = numInteractions + 1 + 2 * inDim1;
    int numElements = inDim1 * inDim0;
    int blocksPerGrid = (numElements + DEFAULT_THREADS_PER_BLOCK - 1) / DEFAULT_THREADS_PER_BLOCK;
    //cudaProfilerStart();

    //Multiply by D1.
    genExactQuadFeatures<T><<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(inArray, outArray, 
                                 inDim1, outDim1, numElements);
    

    //cudaProfilerStop();
    return "no_error";
}
//Instantiate templates explicitly so wrapper can use.
template const char *cudaExactQuadratic_<float>(float inArray[], double *outArray, 
                    int inDim0, int inDim1);
template const char *cudaExactQuadratic_<double>(double inArray[], double *outArray, 
                    int inDim0, int inDim1);




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
    
    polyMultByDiagRademMat<T><<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(copyBuffer,
            radem + numElementsPerRow, numElementsPerRow, numElements, normConstant);
    cudaHTransform3d<T>(copyBuffer, reshapedDim0, reshapedDim1, reshapedDim2);

    polyMultByDiagRademMat<T><<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(copyBuffer,
            radem + 2 * numElementsPerRow, numElementsPerRow, numElements, normConstant);
    cudaHTransform3d<T>(copyBuffer, reshapedDim0, reshapedDim1, reshapedDim2);
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
    
        polyMultByDiagRademMat<T><<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(copyBuffer,
            radem + (i * 3 + 1) * numElementsPerRow,
            numElementsPerRow, numElements, normConstant);
        cudaHTransform3d<T>(copyBuffer, reshapedDim0, reshapedDim1, reshapedDim2);

        polyMultByDiagRademMat<T><<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(copyBuffer,
            radem + (i * 3 + 2) * numElementsPerRow,
            numElementsPerRow, numElements, normConstant);
        cudaHTransform3d<T>(copyBuffer, reshapedDim0, reshapedDim1, reshapedDim2);
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
