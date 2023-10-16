/*
* Contains all functions needed to generate exact quadratic polynomial
* features on GPU.
*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "polynomial_operations.h"

#define DEFAULT_THREADS_PER_BLOCK 256




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
