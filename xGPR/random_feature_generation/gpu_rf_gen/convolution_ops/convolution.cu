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


//Performs a strided copy from the original xdata into a temporary copy
//buffer to set it up for FHT-based convolution.
template <typename T>
__global__ void cudaConvMaxpoolStridedCopyOp(const T xdata[], T copyBuffer[],
            int dim1, int dim2, int numKmers, int bufferDim2, int convWidth,
            int numElements){
    int zLoc = blockIdx.z * blockDim.x + threadIdx.x;

    int outputElement = blockIdx.x * numKmers * bufferDim2;
    outputElement += blockIdx.y * bufferDim2 + zLoc;
    int kmerWidth = convWidth * dim2;

    int inputElement = blockIdx.x * dim1 * dim2;
    inputElement += blockIdx.y * dim2 + zLoc;

    if (outputElement < numElements && (zLoc < kmerWidth))
        copyBuffer[outputElement] = xdata[inputElement];
}


//Performs the final steps in feature generation for maxpool-based convolution
//kernels -- multiplying by chiArr, taking sine or cosine and adding
//to the appropriate elements of outputArray.
template <typename T>
__global__ void conv1dMaxpoolPostProcessKernel(const T featureArray[], const T chiArr[],
            double *outputArray, int dim1, int dim2, int numFreqs,
            int startPosition, int endPosition, int convWidth,
            const int32_t *seqlengths){

    int colCutoff = seqlengths[blockIdx.x] - convWidth + 1;
    int zLoc = blockIdx.y * blockDim.x + threadIdx.x;
    int inputLoc = blockIdx.x * dim1 * dim2 + zLoc;
    int outputLoc = blockIdx.x * numFreqs + zLoc + startPosition;
    const T *featurePtr = featureArray + inputLoc;
    double chiProd, outputVal = 0;

    if (zLoc < endPosition){
        T chiVal = chiArr[startPosition + zLoc];

        for (int i=0; i < (colCutoff * dim2); i+=(gridDim.y * blockDim.x)){
            chiProd = chiVal * featurePtr[i];
            outputVal = MAX(chiProd, outputVal);
        }
        outputArray[outputLoc] = outputVal;
    }
}




//This function generates and sums random features for a Conv1d Maxpool-type kernel.
template <typename T>
const char *conv1dMaxpoolFeatureGen(const int8_t *radem, const T xdata[],
            const T chiArr[], double *outputArray, const int32_t *seqlengths,
            int xdim0, int xdim1, int xdim2, int numFreqs,
            int convWidth, int paddedBufferSize){

    int numKmers = xdim1 - convWidth + 1;
    int numElements = xdim0 * numKmers * paddedBufferSize;

    T *featureArray;
    if (cudaMalloc(&featureArray, sizeof(T) * numElements) != cudaSuccess) {
            cudaFree(featureArray);
            return "Fatal malloc error";
    };

    //This is the Hadamard normalization constant.
    T normConstant = log2(paddedBufferSize) / 2;
    normConstant = 1 / pow(2, normConstant);

    int startPosition, endPosition;
    int thPerCopyBlock = DEFAULT_THREADS_PER_BLOCK, thPerSumBlock;
    int numRepeats = (numFreqs + paddedBufferSize - 1) / paddedBufferSize;
    const char *errCode;

    if (xdim2 <= 64)
        thPerCopyBlock = 64;
    else if (xdim2 < 256)
        thPerCopyBlock = 128;

    //Can do this, because paddedBufferSize is always a power of 2.
    thPerSumBlock = MIN(256, paddedBufferSize);

    dim3 copyBlocks = dim3(xdim0, numKmers, (xdim2 * convWidth + thPerCopyBlock - 1) / thPerCopyBlock);
    dim3 sumBlocks = dim3(xdim0, paddedBufferSize / thPerSumBlock, 1);


    for (int i=0; i < numRepeats; i++){
        startPosition = i * paddedBufferSize;
        endPosition = MIN((i + 1) * paddedBufferSize, numFreqs) - startPosition;

        cudaMemset(featureArray, 0, sizeof(T) * numElements);
        cudaConvMaxpoolStridedCopyOp<T><<<copyBlocks, thPerCopyBlock>>>(xdata, featureArray,
            xdim1, xdim2, numKmers, paddedBufferSize, convWidth, numElements);

        errCode = cudaConvSORF3d<T>(featureArray, radem,
                xdim0, numKmers, paddedBufferSize, startPosition, numElements,
                numFreqs, normConstant);


        //Multiply by chiArr; take the sine and cosine of elements of
        //featureArray, multiply by scalingTerm, and transfer to outputArray.
        conv1dMaxpoolPostProcessKernel<T><<<sumBlocks, thPerSumBlock>>>(featureArray, chiArr,
                outputArray, numKmers, paddedBufferSize, numFreqs,
                startPosition, endPosition, convWidth, seqlengths);
    }

    cudaFree(featureArray);
    return errCode;
}
//Explicitly instantiate so wrapper can use.
template const char *conv1dMaxpoolFeatureGen<float>(const int8_t *radem, const float *xdata,
            const float *chiArr, double *outputArray, const int32_t *seqlengths,
            int xdim0, int xdim1, int xdim2, int numFreqs,
            int convWidth, int paddedBufferSize);
template const char *conv1dMaxpoolFeatureGen<double>(const int8_t *radem, const double *xdata,
            const double *chiArr, double *outputArray, const int32_t *seqlengths,
            int xdim0, int xdim1, int xdim2, int numFreqs,
            int convWidth, int paddedBufferSize);



//This function performs the SORF block transform (HD3 HD2 HD1)
//on input that has been reshaped as appropriate for a convolution,
//without any subsequent modifications. This is useful if the wrapper
//is performing the subsequent steps involved in feature generation.
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
