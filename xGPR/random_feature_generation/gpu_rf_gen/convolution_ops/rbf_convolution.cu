/*
* Contains routines needed specifically for generating features for RBF-based
* convolution kernels (FHTConv1d, GraphConv) and calculating their gradients.
*/
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#include "../basic_ops/basic_array_operations.h"
#include "rbf_convolution.h"

#define DEFAULT_THREADS_PER_BLOCK 256
#define DEFAULT_THREADS_PER_BLREDUCE 32




//Performs a strided copy from the original xdata into a temporary copy
//buffer to set it up for FHT-based convolution.
template <typename T>
__global__ void cudaConvRBFStridedCopyOp(const T xdata[], T copyBuffer[],
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


//Performs the final steps in feature generation for RBF-based convolution
//kernels -- multiplying by chiArr, taking sine or cosine and adding
//to the appropriate elements of outputArray.
template <typename T>
__global__ void convRBFPostProcessKernel(const T featureArray[], const T chiArr[],
            double *outputArray, int dim1, int dim2, int numFreqs,
            int startPosition, int endPosition, double scalingTerm,
            int convWidth, const int32_t *seqlengths){

    int colCutoff = seqlengths[blockIdx.x] - convWidth + 1;
    int zLoc = blockIdx.y * blockDim.x + threadIdx.x;
    int inputLoc = blockIdx.x * dim1 * dim2 + zLoc;
    int outputLoc = blockIdx.x * 2 * numFreqs + 2 * zLoc + 2 * startPosition;
    const T *featurePtr = featureArray + inputLoc;
    double chiProd, sinSum = 0, cosSum = 0;

    if (zLoc < endPosition){
        T chiVal = chiArr[startPosition + zLoc];

        for (int i=0; i < (colCutoff * dim2); i+=(gridDim.y * blockDim.x)){
            chiProd = chiVal * featurePtr[i];
            cosSum += cos(chiProd);
            sinSum += sin(chiProd);
        }
        outputArray[outputLoc] = cosSum * scalingTerm;
        outputArray[outputLoc + 1] = sinSum * scalingTerm;
    }
}



//Performs the final steps in feature generation for RBF-based convolution
//kernels -- multiplying by chiArr, taking sine or cosine and adding
//to the appropriate elements of outputArray.
template <typename T>
__global__ void convRBFGradProcessKernel(const T featureArray[], const T chiArr[],
            double *outputArray, int dim1, int dim2, int numFreqs,
            int startPosition, int endPosition, double scalingTerm,
            double sigma, double *gradientArray,
            int convWidth, const int32_t *seqlengths){

    int colCutoff = seqlengths[blockIdx.x] - convWidth + 1;
    int zLoc = blockIdx.y * blockDim.x + threadIdx.x;
    int inputLoc = blockIdx.x * dim1 * dim2 + zLoc;
    int outputLoc = blockIdx.x * 2 * numFreqs + 2 * zLoc + 2 * startPosition;
    const T *featurePtr = featureArray + inputLoc;
    double chiProd, cosVal, sinVal, sinSum = 0, cosSum = 0;
    double gradSinVal = 0, gradCosVal = 0;

    if (zLoc < endPosition){
        T chiVal = chiArr[startPosition + zLoc];

        for (int i=0; i < (colCutoff * dim2); i+=(gridDim.y * blockDim.x)){
            chiProd = chiVal * featurePtr[i];
            cosVal = cos(chiProd * sigma);
            sinVal = sin(chiProd * sigma);

            cosSum += cosVal;
            sinSum += sinVal;
            gradCosVal -= sinVal * chiProd;
            gradSinVal += cosVal * chiProd;
        }

        outputArray[outputLoc] = cosSum * scalingTerm;
        outputArray[outputLoc + 1] = sinSum * scalingTerm;
        gradientArray[outputLoc] = gradCosVal * scalingTerm;
        gradientArray[outputLoc + 1] = gradSinVal * scalingTerm;
    }
}



//This function generates and sums random features for an
//input array reshapedX of input type float.
template <typename T>
const char *convRBFFeatureGen(const int8_t *radem, const T xdata[],
            const T chiArr[], double *outputArray, const int32_t *seqlengths,
            int xdim0, int xdim1, int xdim2, int numFreqs,
            int rademShape2, int convWidth, int paddedBufferSize,
            double scalingTerm){

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
        cudaConvRBFStridedCopyOp<T><<<copyBlocks, thPerCopyBlock>>>(xdata, featureArray,
            xdim1, xdim2, numKmers, paddedBufferSize, convWidth, numElements);

        errCode = cudaConvSORF3d<T>(featureArray, radem,
                xdim0, numKmers, paddedBufferSize, startPosition, numElements,
                rademShape2, normConstant);


        //Multiply by chiArr; take the sine and cosine of elements of
        //featureArray, multiply by scalingTerm, and transfer to outputArray.
        convRBFPostProcessKernel<T><<<sumBlocks, thPerSumBlock>>>(featureArray, chiArr,
                outputArray, numKmers, paddedBufferSize, numFreqs,
                startPosition, endPosition, scalingTerm,
                convWidth, seqlengths);
    }

    cudaFree(featureArray);
    return errCode;
}
//Explicitly instantiate so wrapper can use.
template const char *convRBFFeatureGen<float>(const int8_t *radem,
            const float xdata[], const float chiArr[], double *outputArray,
            const int32_t *seqlengths, int xdim0, int xdim1, int xdim2, int numFreqs,
            int rademShape2, int convWidth, int paddedBufferSize,
            double scalingTerm);
template const char *convRBFFeatureGen<double>(const int8_t *radem,
            const double xdata[], const double chiArr[], double *outputArray,
            const int32_t *seqlengths, int xdim0, int xdim1, int xdim2, int numFreqs,
            int rademShape2, int convWidth, int paddedBufferSize,
            double scalingTerm);



//This function generates and sums random features for an
//input array reshapedX of input type float WHILE also
//generating gradient information and storing this in
//a separate array. This gradient is only applicable
//in cases where all of the features share the same
//lengthscale; ARD-type kernels require a more complicated
//gradient calculation not implemented here.
template <typename T>
const char *convRBFFeatureGrad(const int8_t *radem, const T xdata[],
            const T chiArr[], double *outputArray, const int32_t *seqlengths,
            double *gradientArray, double sigma,
            int xdim0, int xdim1, int xdim2, int numFreqs,
            int rademShape2, int convWidth, int paddedBufferSize,
            double scalingTerm){

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
        cudaConvRBFStridedCopyOp<T><<<copyBlocks, thPerCopyBlock>>>(xdata, featureArray,
            xdim1, xdim2, numKmers, paddedBufferSize, convWidth, numElements);

        errCode = cudaConvSORF3d<T>(featureArray, radem,
                xdim0, numKmers, paddedBufferSize,
                startPosition, numElements, rademShape2, normConstant);

        //Multiply by chiArr; take the sine and cosine of elements of
        //featureArray, multiply by scalingTerm, transfer to output
        //AND at the same time calculate the gradient terms, using
        //them to populate gradientArray.
        convRBFGradProcessKernel<T><<<sumBlocks, thPerSumBlock>>>(featureArray, chiArr,
                outputArray, numKmers, paddedBufferSize, numFreqs,
                startPosition, endPosition, scalingTerm, sigma,
                gradientArray, convWidth, seqlengths);
    }

    cudaFree(featureArray);
    return errCode;
}
//Explicitly instantiate so wrapper can use.
template const char *convRBFFeatureGrad<float>(const int8_t *radem,
            const float xdata[], const float chiArr[], double *outputArray,
            const int32_t *seqlengths, double *gradientArray, double sigma,
            int xdim0, int xdim1, int xdim2, int numFreqs,
            int rademShape2, int convWidth, int paddedBufferSize,
            double scalingTerm);
template const char *convRBFFeatureGrad<double>(const int8_t *radem,
            const double xdata[], const double chiArr[], double *outputArray,
            const int32_t *seqlengths, double *gradientArray, double sigma,
            int xdim0, int xdim1, int xdim2, int numFreqs,
            int rademShape2, int convWidth, int paddedBufferSize,
            double scalingTerm);
