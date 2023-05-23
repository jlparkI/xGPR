/*
* Contains routines needed specifically for generating features for ArcCos-based
* convolution kernels and calculating their gradients.
*/
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#include "../basic_ops/float_array_operations.h"
#include "../basic_ops/double_array_operations.h"
#include "arccos_convolution.h"

#define DEFAULT_THREADS_PER_BLOCK 256
#define DEFAULT_THREADS_PER_BLREDUCE 32




//Performs an elementwise multiplication of a row of a [3,1,P x S] array against the
//float [N,M,S] input array. Note that the dimensions must be checked before calling
//-- done by the wrapper -- and that only S elements of the appropriate row of
//the [3, 1, P x S] array are used.
__global__ void floatConv1dArcCosRademMultiply(float *cArray,
            const int8_t *rademArray,
			int dim2, int startPosition, int numElements, float normConstant)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int position = startPosition + (tid & (dim2 - 1));
    
    if (tid < numElements)
        cArray[tid] = cArray[tid] * rademArray[position] * normConstant;
}



//Performs an elementwise multiplication of a row of a [3,1,P x S] array against the
//double [N,M,S] input array. Note that the dimensions must be checked before calling
//-- done by the wrapper -- and that only S elements of the appropriate row of
//the [3, 1, P x S] array are used.
__global__ void doubleConv1dArcCosRademMultiply(double *cArray,
            const int8_t *rademArray,
			int dim2, int startPosition, int numElements, double normConstant)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int position = startPosition + (tid & (dim2 - 1));
    
    if (tid < numElements)
        cArray[tid] = cArray[tid] * rademArray[position] * normConstant;
}



//Performs an elementwise multiplication by a diagonal matrix populated with
//elements from a Rademacher distribution, while also multiplying by the
//Hadamard norm constant and copying into the featureArray array.
__global__ void floatConv1dArcCosRademAndCopy(const float *inputArray, float *featureArray,
            const int8_t *rademArray, int dim2, int startPosition,
            int numElements, float normConstant)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int position = startPosition + (tid & (dim2 - 1));
    
    if (tid < numElements)
        featureArray[tid] = inputArray[tid] * rademArray[position] * normConstant;
}


//Performs an elementwise multiplication by a diagonal matrix populated with
//elements from a Rademacher distribution, while also multiplying by the
//Hadamard norm constant and copying into the featureArray array.
__global__ void doubleConv1dArcCosRademAndCopy(const double *inputArray, double *featureArray,
            const int8_t *rademArray, int dim2, int startPosition,
            int numElements, double normConstant)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int position = startPosition + (tid & (dim2 - 1));
    
    if (tid < numElements)
        featureArray[tid] = inputArray[tid] * rademArray[position] * normConstant;
}




//Performs the final steps in feature generation for ArcCos-based convolution
//kernels -- multiplying by chiArr, taking sine or cosine and adding
//to the appropriate elements of outputArray.
__global__ void floatConvArcCosPostProcessKernelOrder1(const float *featureArray, float *chiArr,
            double *outputArray, int dim1, int dim2, int numFreqs,
            int startPosition, int numElements,
            int endPosition, double scalingTerm){
    int i;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int column = tid % endPosition;
    int row = tid / endPosition;
    int inputLoc = row * dim1 * dim2 + column;
    int outputLoc = row * numFreqs + column + startPosition;
    float *chiVal = chiArr + startPosition + column;
    float chiProd, rollingSum = 0;

    if (tid < numElements){
        for (i=0; i < dim1; i++){
            chiProd = *chiVal * featureArray[inputLoc];
            rollingSum += max(chiProd, 0.0);
            inputLoc += dim2;
        }
        outputArray[outputLoc] = rollingSum * scalingTerm;
    }
}



//Performs the final steps in feature generation for ArcCos-based convolution
//kernels -- multiplying by chiArr, taking sine or cosine and adding
//to the appropriate elements of outputArray.
__global__ void doubleConvArcCosPostProcessKernelOrder1(const double *featureArray, double *chiArr,
            double *outputArray, int dim1, int dim2, int numFreqs,
            int startPosition, int numElements,
            int endPosition, double scalingTerm)
{
    int i;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int column = tid % endPosition;
    int row = tid / endPosition;
    int inputLoc = row * dim1 * dim2 + column;
    int outputLoc = row * numFreqs + column + startPosition;
    double *chiVal = chiArr + startPosition + column;
    double chiProd, rollingSum = 0;

    if (tid < numElements){
        for (i=0; i < dim1; i++){
            chiProd = *chiVal * featureArray[inputLoc];
            rollingSum += max(chiProd, 0.0);
            inputLoc += dim2;
        }
        outputArray[outputLoc] = rollingSum * scalingTerm;
    }
}



//Performs the final steps in feature generation for ArcCos-based convolution
//kernels -- multiplying by chiArr, taking sine or cosine and adding
//to the appropriate elements of outputArray.
__global__ void floatConvArcCosPostProcessKernelOrder2(const float *featureArray, float *chiArr,
            double *outputArray, int dim1, int dim2, int numFreqs,
            int startPosition, int numElements,
            int endPosition, double scalingTerm){
    int i;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int column = tid % endPosition;
    int row = tid / endPosition;
    int inputLoc = row * dim1 * dim2 + column;
    int outputLoc = row * numFreqs + column + startPosition;
    float *chiVal = chiArr + startPosition + column;
    float chiProd, rollingSum = 0;

    if (tid < numElements){
        for (i=0; i < dim1; i++){
            chiProd = *chiVal * featureArray[inputLoc];
            chiProd = max(chiProd, 0.0);
            rollingSum += chiProd * chiProd;
            inputLoc += dim2;
        }
        outputArray[outputLoc] = rollingSum * scalingTerm;
    }
}



//Performs the final steps in feature generation for ArcCos-based convolution
//kernels -- multiplying by chiArr, taking sine or cosine and adding
//to the appropriate elements of outputArray.
__global__ void doubleConvArcCosPostProcessKernelOrder2(const double *featureArray, double *chiArr,
            double *outputArray, int dim1, int dim2, int numFreqs,
            int startPosition, int numElements,
            int endPosition, double scalingTerm)
{
    int i;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int column = tid % endPosition;
    int row = tid / endPosition;
    int inputLoc = row * dim1 * dim2 + column;
    int outputLoc = row * numFreqs + column + startPosition;
    double *chiVal = chiArr + startPosition + column;
    double chiProd, rollingSum = 0;

    if (tid < numElements){
        for (i=0; i < dim1; i++){
            chiProd = *chiVal * featureArray[inputLoc];
            chiProd = max(chiProd, 0.0);
            rollingSum += chiProd * chiProd;
            inputLoc += dim2;
        }
        outputArray[outputLoc] = rollingSum * scalingTerm;
    }
}



//This function generates and sums random features for an
//input array reshapedX of input type float.
const char *floatConvArcCosFeatureGen(int8_t *radem, float *reshapedX,
            float *featureArray, float *chiArr, double *outputArray,     
            int reshapedDim0, int reshapedDim1, int reshapedDim2,
            int numFreqs, int rademShape2, double scalingTerm,
            int kernelOrder){

    int numElements = reshapedDim0 * reshapedDim1 * reshapedDim2;
    //This is the Hadamard normalization constant.
    float normConstant = log2(reshapedDim2) / 2;
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
        floatConv1dArcCosRademAndCopy<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(reshapedX, 
                        featureArray, radem, reshapedDim2, startPosition, numElements,
                        normConstant);
        //First H-transform.
        floatCudaHTransform3d(featureArray, reshapedDim0, reshapedDim1, reshapedDim2);

        //Multiply by second row of radem.
        floatConv1dArcCosRademMultiply<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(featureArray, 
                        radem + rademShape2, reshapedDim2, startPosition, numElements,
                        normConstant);
        //Second H-transform.
        floatCudaHTransform3d(featureArray, reshapedDim0, reshapedDim1, reshapedDim2);
        
        //Multiply by third row of radem.
        floatConv1dArcCosRademMultiply<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(featureArray, 
                        radem + 2 * rademShape2, reshapedDim2, startPosition, numElements,
                        normConstant);
        //Last H-transform.
        floatCudaHTransform3d(featureArray, reshapedDim0, reshapedDim1, reshapedDim2);


        //Multiply by chiArr; take the sine and cosine of elements of
        //featureArray, multiply by scalingTerm, and transfer to outputArray.
        if (kernelOrder == 1){
            floatConvArcCosPostProcessKernelOrder1<<<outBlocks, DEFAULT_THREADS_PER_BLOCK>>>(featureArray, chiArr,
                outputArray, reshapedDim1, reshapedDim2, numFreqs,
                startPosition, numOutElements, endPosition,
                scalingTerm);
        }
        else{
            floatConvArcCosPostProcessKernelOrder2<<<outBlocks, DEFAULT_THREADS_PER_BLOCK>>>(featureArray, chiArr,
                outputArray, reshapedDim1, reshapedDim2, numFreqs,
                startPosition, numOutElements, endPosition,
                scalingTerm);
        }
    }

    return "no_error";
}



//This function generates and sums random features for an
//input array reshapedX of input type double.
const char *doubleConvArcCosFeatureGen(int8_t *radem, double *reshapedX,
                double *featureArray, double *chiArr, double *outputArray,
                int reshapedDim0, int reshapedDim1, int reshapedDim2,
                int numFreqs, int rademShape2, double scalingTerm,
                int kernelOrder){

    int numElements = reshapedDim0 * reshapedDim1 * reshapedDim2;
    //This is the Hadamard normalization constant.
    double normConstant = log2(reshapedDim2) / 2;
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
        doubleConv1dArcCosRademAndCopy<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(reshapedX, 
                        featureArray, radem, reshapedDim2, startPosition, numElements,
                        normConstant);
        //First H-transform.
        doubleCudaHTransform3d(featureArray, reshapedDim0, reshapedDim1, reshapedDim2);

        //Multiply by second row of radem.
        doubleConv1dArcCosRademMultiply<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(featureArray, 
                        radem + rademShape2, reshapedDim2, startPosition, numElements,
                        normConstant);
        //Second H-transform.
        doubleCudaHTransform3d(featureArray, reshapedDim0, reshapedDim1, reshapedDim2);
        
        //Multiply by third row of radem.
        doubleConv1dArcCosRademMultiply<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(featureArray, 
                        radem + 2 * rademShape2, reshapedDim2, startPosition, numElements,
                        normConstant);
        //Last H-transform.
        doubleCudaHTransform3d(featureArray, reshapedDim0, reshapedDim1, reshapedDim2);

        //Multiply by chiArr; take the sine and cosine of elements of
        //featureArray, multiply by scalingTerm, and transfer to output.
        if (kernelOrder == 1){
            doubleConvArcCosPostProcessKernelOrder1<<<outBlocks, DEFAULT_THREADS_PER_BLOCK>>>(featureArray, chiArr,
                outputArray, reshapedDim1, reshapedDim2, numFreqs,
                startPosition, numOutElements, endPosition,
                scalingTerm);
        }
        else{
            doubleConvArcCosPostProcessKernelOrder2<<<outBlocks, DEFAULT_THREADS_PER_BLOCK>>>(featureArray, chiArr,
                outputArray, reshapedDim1, reshapedDim2, numFreqs,
                startPosition, numOutElements, endPosition,
                scalingTerm);
        }
    }

    return "no_error";
}
