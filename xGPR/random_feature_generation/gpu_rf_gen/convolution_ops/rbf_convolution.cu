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




//Performs an elementwise multiplication of a row of a [3,1,P x S] array against the
//float [N,M,S] input array. Note that the dimensions must be checked before calling
//-- done by the wrapper -- and that only S elements of the appropriate row of
//the [3, 1, P x S] array are used.
template <typename T>
__global__ void conv1dRBFRademMultiply(T cArray[],
            const int8_t *rademArray,
			int dim2, int startPosition, int numElements,
            T normConstant)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int position = startPosition + (tid & (dim2 - 1));
    
    if (tid < numElements)
        cArray[tid] = cArray[tid] * rademArray[position] * normConstant;
}

//Performs an elementwise multiplication by a diagonal matrix populated with
//elements from a Rademacher distribution, while also multiplying by the
//Hadamard norm constant and copying into the featureArray array.
template <typename T>
__global__ void conv1dRBFRademAndCopy(T inputArray[], T featureArray[],
            const int8_t *rademArray, int dim2, int startPosition,
            int numElements, T normConstant)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int position = startPosition + (tid & (dim2 - 1));
    
    if (tid < numElements)
        featureArray[tid] = inputArray[tid] * rademArray[position] * normConstant;
}




//Performs the final steps in feature generation for RBF-based convolution
//kernels -- multiplying by chiArr, taking sine or cosine and adding
//to the appropriate elements of outputArray.
template <typename T>
__global__ void convRBFPostProcessKernel(T featureArray[], T chiArr[],
            double *outputArray, int dim1, int dim2, int numFreqs,
            int startPosition, int numElements,
            int endPosition, double scalingTerm){
    int i;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int column = tid % endPosition;
    int row = tid / endPosition;
    int inputLoc = row * dim1 * dim2 + column;
    int outputLoc = row * 2 * numFreqs + 2 * column + 2 * startPosition;
    T chiVal = chiArr[startPosition + column];
    double chiProd, sinSum = 0, cosSum = 0;

    if (tid < numElements){
        for (i=0; i < dim1; i++){
            chiProd = chiVal * featureArray[inputLoc];
            cosSum += cos(chiProd);
            sinSum += sin(chiProd);
            inputLoc += dim2;
        }
        outputArray[outputLoc] = cosSum * scalingTerm;
        outputArray[outputLoc + 1] = sinSum * scalingTerm;
    }
}


//Performs the final steps in feature generation WITH simultaneous gradient
//calculation for RBF-based convolution
//kernels -- multiplying by chiArr, taking sine or cosine and adding
//to the appropriate elements of outputArray.
template <typename T>
__global__ void convRBFGradProcessKernel(T featureArray[], T chiArr[],
            double *outputArray, int dim1, int dim2, int numFreqs,
            int startPosition, int numElements,
            int endPosition, double scalingTerm,
            double sigma, double *gradientArray)
{
    int i;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int column = tid % endPosition;
    int row = tid / endPosition;
    int inputLoc = row * dim1 * dim2 + column;
    int outputLoc = row * 2 * numFreqs + 2 * column + 2 * startPosition;
    T chiVal = chiArr[startPosition + column];
    double chiProd, sinSum = 0, cosSum = 0, sinVal, cosVal;
    double gradSinVal = 0, gradCosVal = 0;

    if (tid < numElements){
        for (i=0; i < dim1; i++){
            chiProd = chiVal * featureArray[inputLoc];
            cosVal = cos(chiProd * sigma);
            sinVal = sin(chiProd * sigma);

            cosSum += cosVal;
            sinSum += sinVal;
            //These are the derivatives.
            gradCosVal -= sinVal * chiProd;
            gradSinVal += cosVal * chiProd;
            inputLoc += dim2;
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
const char *convRBFFeatureGen(int8_t *radem, T reshapedX[],
            T featureArray[], T chiArr[], double *outputArray,     
            int reshapedDim0, int reshapedDim1, int reshapedDim2,
            int numFreqs, int rademShape2, double scalingTerm){

    int numElements = reshapedDim0 * reshapedDim1 * reshapedDim2;
    //This is the Hadamard normalization constant.
    T normConstant = log2(reshapedDim2) / 2;
    normConstant = 1 / pow(2, normConstant);
    int blocksPerGrid = (numElements + DEFAULT_THREADS_PER_BLOCK - 1) / 
                DEFAULT_THREADS_PER_BLOCK;

    int endPosition, numOutElements, outBlocks;
    int numRepeats = (numFreqs + reshapedDim2 - 1) / reshapedDim2;
    int i, startPosition;

    for (i=0; i < numRepeats; i++){
        startPosition = i * reshapedDim2;
        endPosition = MIN((i + 1) * reshapedDim2, numFreqs);
        endPosition -= i * reshapedDim2;
        numOutElements = reshapedDim0 * endPosition;
        outBlocks = (numOutElements + DEFAULT_THREADS_PER_BLOCK - 1) / DEFAULT_THREADS_PER_BLOCK;

        //Copy input into featureArray while multiplying by first row of radem.
        conv1dRBFRademAndCopy<T><<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(reshapedX, 
                        featureArray, radem, reshapedDim2, startPosition, numElements,
                        normConstant);
        //First H-transform.
        cudaHTransform3d<T>(featureArray, reshapedDim0, reshapedDim1, reshapedDim2);

        //Multiply by second row of radem.
        conv1dRBFRademMultiply<T><<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(featureArray, 
                        radem + rademShape2, reshapedDim2, startPosition, numElements,
                        normConstant);
        //Second H-transform.
        cudaHTransform3d<T>(featureArray, reshapedDim0, reshapedDim1, reshapedDim2);
        
        //Multiply by third row of radem.
        conv1dRBFRademMultiply<T><<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(featureArray, 
                        radem + 2 * rademShape2, reshapedDim2, startPosition, numElements,
                        normConstant);
        //Last H-transform.
        cudaHTransform3d<T>(featureArray, reshapedDim0, reshapedDim1, reshapedDim2);


        //Multiply by chiArr; take the sine and cosine of elements of
        //featureArray, multiply by scalingTerm, and transfer to outputArray.
        convRBFPostProcessKernel<T><<<outBlocks, DEFAULT_THREADS_PER_BLOCK>>>(featureArray, chiArr,
                outputArray, reshapedDim1, reshapedDim2, numFreqs,
                startPosition, numOutElements, endPosition,
                scalingTerm);
    }

    return "no_error";
}
//Explicitly instantiate so wrapper can use.
template const char *convRBFFeatureGen<float>(int8_t *radem, float reshapedX[],
            float featureArray[], float chiArr[], double *outputArray,     
            int reshapedDim0, int reshapedDim1, int reshapedDim2,
            int numFreqs, int rademShape2, double scalingTerm);
template const char *convRBFFeatureGen<double>(int8_t *radem, double reshapedX[],
            double featureArray[], double chiArr[], double *outputArray,     
            int reshapedDim0, int reshapedDim1, int reshapedDim2,
            int numFreqs, int rademShape2, double scalingTerm);



//This function generates and sums random features for an
//input array reshapedX of input type float WHILE also
//generating gradient information and storing this in
//a separate array. This gradient is only applicable
//in cases where all of the features share the same
//lengthscale; ARD-type kernels require a more complicated
//gradient calculation not implemented here.
template <typename T>
const char *convRBFFeatureGrad(int8_t *radem, T reshapedX[],
            T featureArray[], T chiArr[], double *outputArray,     
            double *gradientArray, double sigma,
            int reshapedDim0, int reshapedDim1, int reshapedDim2,
            int numFreqs, int rademShape2, double scalingTerm){

    int numElements = reshapedDim0 * reshapedDim1 * reshapedDim2;
    //This is the Hadamard normalization constant.
    T normConstant = log2(reshapedDim2) / 2;
    normConstant = 1 / pow(2, normConstant);
    int blocksPerGrid = (numElements + DEFAULT_THREADS_PER_BLOCK - 1) / 
                DEFAULT_THREADS_PER_BLOCK;
    int numOutElements, outBlocks;

    int numRepeats = (numFreqs + reshapedDim2 - 1) / reshapedDim2;
    int i, startPosition, endPosition;

    for (i=0; i < numRepeats; i++){
        startPosition = i * reshapedDim2;
        endPosition = MIN((i + 1) * reshapedDim2, numFreqs);
        endPosition -= startPosition;
        numOutElements = reshapedDim0 * endPosition;
        outBlocks = (numOutElements + DEFAULT_THREADS_PER_BLOCK - 1) / DEFAULT_THREADS_PER_BLOCK;

        //Copy input into featureArray while multiplying by first row of radem.
        conv1dRBFRademAndCopy<T><<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(reshapedX, 
                        featureArray, radem, reshapedDim2, startPosition, numElements,
                        normConstant);
        //First H-transform.
        cudaHTransform3d<T>(featureArray, reshapedDim0, reshapedDim1, reshapedDim2);

        //Multiply by second row of radem.
        conv1dRBFRademMultiply<T><<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(featureArray, 
                        radem + rademShape2, reshapedDim2, startPosition, numElements,
                        normConstant);
        //Second H-transform.
        cudaHTransform3d<T>(featureArray, reshapedDim0, reshapedDim1, reshapedDim2);
        
        //Multiply by third row of radem.
        conv1dRBFRademMultiply<T><<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(featureArray, 
                        radem + 2 * rademShape2, reshapedDim2, startPosition, numElements,
                        normConstant);
        //Last H-transform.
        cudaHTransform3d<T>(featureArray, reshapedDim0, reshapedDim1, reshapedDim2);

        //Multiply by chiArr; take the sine and cosine of elements of
        //featureArray, multiply by scalingTerm, transfer to output
        //AND at the same time calculate the gradient terms, using
        //them to populate gradientArray.
        convRBFGradProcessKernel<T><<<outBlocks, DEFAULT_THREADS_PER_BLOCK>>>(featureArray, chiArr,
                outputArray, reshapedDim1, reshapedDim2, numFreqs,
                startPosition, numOutElements, endPosition,
                scalingTerm, sigma, gradientArray);
    }

    return "no_error";
}
//Explicitly instantiate so wrapper can use.
template const char *convRBFFeatureGrad<float>(int8_t *radem, float reshapedX[],
            float featureArray[], float chiArr[], double *outputArray,     
            double *gradientArray, double sigma,
            int reshapedDim0, int reshapedDim1, int reshapedDim2,
            int numFreqs, int rademShape2, double scalingTerm);
template const char *convRBFFeatureGrad<double>(int8_t *radem, double reshapedX[],
            double featureArray[], double chiArr[], double *outputArray,     
            double *gradientArray, double sigma,
            int reshapedDim0, int reshapedDim1, int reshapedDim2,
            int numFreqs, int rademShape2, double scalingTerm);
