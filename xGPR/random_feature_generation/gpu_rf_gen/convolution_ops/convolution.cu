/*
* Contains all functions needed to run the structured orthogonal features
* (SORF) operation on an input 3d array that has been restructured for
* convolution on GPU. The input array should already live on GPU.
* The Hadamard transforms are performed using functions from float_array_operations.cu
* and double_array_operations.cu, the diagonal matrix multiplication is slightly
* different and so is implemented here.
*/
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cub/cub.cuh>
#include "../float_array_operations.h"
#include "../double_array_operations.h"
#include "convolution.h"

#define DEFAULT_THREADS_PER_BLOCK 256
#define DEFAULT_THREADS_PER_BLREDUCE 32




//Performs an elementwise multiplication of a row of a [3,1,P x S] array against the
//float [N,M,S] input array. Note that the dimensions must be checked before calling
//-- done by the wrapper -- and that only S elements of the appropriate row of
//the [3, 1, P x S] array are used.
__global__ void floatConv1dMultiplyByRadem(float *cArray, int8_t *rademArray,
			int dim2, int startPosition, int numElements, float normConstant)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int8_t *rVal = rademArray + startPosition + (j & (dim2 - 1));
    
    if (j < numElements)
        cArray[j] = cArray[j] * *rVal * normConstant;
}



//Performs an elementwise multiplication of a row of a [3,1,P x S] array against the
//double [N,M,S] input array. Note that the dimensions must be checked before calling
//-- done by the wrapper -- and that only S elements of the appropriate row of
//the [3, 1, P x S] array are used.
__global__ void doubleConv1dMultiplyByRadem(double *cArray, int8_t *rademArray,
			int dim2, int startPosition, int numElements, double normConstant)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int8_t *rVal = rademArray + startPosition + (j & (dim2 - 1));
    
    if (j < numElements)
        cArray[j] = cArray[j] * *rVal * normConstant;
}



//Performs an elementwise multiplication by a diagonal matrix populated with
//elements from a Rademacher distribution, while also multiplying by the
//Hadamard norm constant and copying into the featureArray array.
__global__ void floatConv1dRademAndCopy(float *inputArray, float *featureArray,
            int8_t *rademArray, int dim2, int startPosition,
            int numElements, float normConstant)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int8_t *rVal = rademArray + startPosition + (j & (dim2 - 1));
    
    if (j < numElements)
        featureArray[j] = inputArray[j] * *rVal * normConstant;
}


//Performs an elementwise multiplication by a diagonal matrix populated with
//elements from a Rademacher distribution, while also multiplying by the
//Hadamard norm constant and copying into the featureArray array.
__global__ void doubleConv1dRademAndCopy(double *inputArray, double *featureArray,
            int8_t *rademArray, int dim2, int startPosition,
            int numElements, double normConstant)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int8_t *rVal = rademArray + startPosition + (j & (dim2 - 1));
    
    if (j < numElements)
        featureArray[j] = inputArray[j] * *rVal * normConstant;
}




//Performs the final steps in feature generation for RBF-based convolution
//kernels -- multiplying by chiArr, taking sine or cosine and adding
//to the appropriate elements of outputArray.
__global__ void floatConvRBFPostProcessKernel(float *featureArray, float *chiArr,
            double *outputArray, int dim1, int dim2,
            int outputDim1, int startPosition, int numElements,
            int log2dim2, double scalingTerm)
{
    int i;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int column = (tid & (dim2 - 1));
    int row = (tid >> log2dim2);
    int inputLoc = row * dim1 * dim2 + column;
    int outputLoc = row * outputDim1 + column + 2 * startPosition;
    float *chiVal = chiArr + startPosition + column;
    float chiProd, sinVal = 0, cosVal = 0;

    if (tid < numElements){
        for (i=0; i < dim1; i++){
            chiProd = *chiVal * featureArray[inputLoc];
            cosVal += cosf(chiProd);
            sinVal += sinf(chiProd);
            inputLoc += dim2;
        }
        outputArray[outputLoc] = cosVal * scalingTerm;
        outputArray[outputLoc + dim2] = sinVal * scalingTerm;
    }
}



//Performs the final steps in feature generation for RBF-based convolution
//kernels -- multiplying by chiArr, taking sine or cosine and adding
//to the appropriate elements of outputArray.
__global__ void doubleConvRBFPostProcessKernel(double *featureArray, double *chiArr,
            double *outputArray, int dim1, int dim2,
            int outputDim1, int startPosition, int numElements,
            int log2dim2, double scalingTerm)
{
    int i;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int column = (tid & (dim2 - 1));
    int row = (tid >> log2dim2);
    int inputLoc = row * dim1 * dim2 + column;
    int outputLoc = row * outputDim1 + column + 2 * startPosition;
    double *chiVal = chiArr + startPosition + column;
    double chiProd, sinVal = 0, cosVal = 0;

    if (tid < numElements){
        for (i=0; i < dim1; i++){
            chiProd = *chiVal * featureArray[inputLoc];
            cosVal += cos(chiProd);
            sinVal += sin(chiProd);
            inputLoc += dim2;
        }
        outputArray[outputLoc] = cosVal * scalingTerm;
        outputArray[outputLoc + dim2] = sinVal * scalingTerm;
    }
}





//This function performs the SORF block transform (HD3 HD2 HD1)
//on float input that has been reshaped as appropriate for a convolution.
//Note that reshapedX must have the same size across the
//last two dimensions as radem and its last dimension must
//be a power of two -- if those conditions are not met, you may
//get an unpredictable result! The Cython wrapper checks all
//of these criteria.
const char *floatConv1dPrep(int8_t *radem, float *reshapedX, int reshapedDim0, 
                int reshapedDim1, int reshapedDim2, int startPosition,
                int numFreqs){

    int numElements = reshapedDim0 * reshapedDim1 * reshapedDim2;
    //This is the Hadamard normalization constant.
    float normConstant = log2(reshapedDim2) / 2;
    normConstant = 1 / pow(2, normConstant);
    int blocksPerGrid = (numElements + DEFAULT_THREADS_PER_BLOCK - 1) / 
                DEFAULT_THREADS_PER_BLOCK;
    
    //Multiply by first row of radem.
    floatConv1dMultiplyByRadem<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(reshapedX, 
                        radem, reshapedDim2, startPosition, numElements,
                        normConstant);
    //First H-transform.
    floatCudaHTransform3d(reshapedX, reshapedDim0, reshapedDim1, reshapedDim2);

    //Multiply by second row of radem.
    floatConv1dMultiplyByRadem<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(reshapedX, 
                        radem + numFreqs, reshapedDim2, startPosition, numElements,
                        normConstant);
    //Second H-transform.
    floatCudaHTransform3d(reshapedX, reshapedDim0, reshapedDim1, reshapedDim2);
        
    //Multiply by third row of radem.
    floatConv1dMultiplyByRadem<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(reshapedX, 
                        radem + 2 * numFreqs, reshapedDim2, startPosition, numElements,
                        normConstant);
    //Last H-transform. Transform is in place so do not need to return anything except no error message.
    floatCudaHTransform3d(reshapedX, reshapedDim0, reshapedDim1, reshapedDim2);
    return "no_error";
}



//This function performs the SORF block transform (HD3 HD2 HD1)
//on double input that has been reshaped as appropriate for a convolution.
//Note that reshapedX must have the same size across the
//last two dimensions as radem and its last dimension must
//be a power of two -- if those conditions are not met, you may
//get an unpredictable result! The Cython wrapper checks all
//of these criteria.
const char *doubleConv1dPrep(int8_t *radem, double *reshapedX, int reshapedDim0, 
                int reshapedDim1, int reshapedDim2, int startPosition,
                int numFreqs){

    int numElements = reshapedDim0 * reshapedDim1 * reshapedDim2;
    //This is the Hadamard normalization constant.
    double normConstant = log2(reshapedDim2) / 2;
    normConstant = 1 / pow(2, normConstant);
    int blocksPerGrid = (numElements + DEFAULT_THREADS_PER_BLOCK - 1) / 
                DEFAULT_THREADS_PER_BLOCK;
    
    //Multiply by first row of radem.
    doubleConv1dMultiplyByRadem<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(reshapedX, 
                        radem, reshapedDim2, startPosition, numElements,
                        normConstant);
    //First H-transform.
    doubleCudaHTransform3d(reshapedX, reshapedDim0, reshapedDim1, reshapedDim2);

    //Multiply by second row of radem.
    doubleConv1dMultiplyByRadem<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(reshapedX, 
                        radem + numFreqs, reshapedDim2, startPosition, numElements,
                        normConstant);
    //Second H-transform.
    doubleCudaHTransform3d(reshapedX, reshapedDim0, reshapedDim1, reshapedDim2);
        
    //Multiply by third row of radem.
    doubleConv1dMultiplyByRadem<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(reshapedX, 
                        radem + 2 * numFreqs, reshapedDim2, startPosition, numElements,
                        normConstant);
    //Last H-transform. Transform is in place so do not need to return anything except no error message.
    doubleCudaHTransform3d(reshapedX, reshapedDim0, reshapedDim1, reshapedDim2);
    return "no_error";
}





//This function generates but does not sum random features for an
//input array reshapedX of input type float. The sine and cosine
//random features are stored in two arrays supplied by caller,
//who can then either sum or further modify as appropriate.
const char *floatConvRBFFeatureGen(int8_t *radem, float *reshapedX,
            float *featureArray, float *chiArr, double *outputArray,     
            int reshapedDim0, int reshapedDim1, int reshapedDim2,
            int numFreqs, double scalingTerm){

    int numElements = reshapedDim0 * reshapedDim1 * reshapedDim2;
    //This is the Hadamard normalization constant.
    float normConstant = log2(reshapedDim2) / 2;
    normConstant = 1 / pow(2, normConstant);
    int blocksPerGrid = (numElements + DEFAULT_THREADS_PER_BLOCK - 1) / 
                DEFAULT_THREADS_PER_BLOCK;

    int numRepeats = (numFreqs + reshapedDim2 - 1) / reshapedDim2;
    int i, startPosition;

    for (i=0; i < numRepeats; i++){
        startPosition = i * reshapedDim2;

        //Copy input into featureArray while multiplying by first row of radem.
        floatConv1dRademAndCopy<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(reshapedX, 
                        featureArray, radem, reshapedDim2, startPosition, numElements,
                        normConstant);
        //First H-transform.
        floatCudaHTransform3d(featureArray, reshapedDim0, reshapedDim1, reshapedDim2);

        //Multiply by second row of radem.
        floatConv1dMultiplyByRadem<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(featureArray, 
                        radem + numFreqs, reshapedDim2, startPosition, numElements,
                        normConstant);
        //Second H-transform.
        floatCudaHTransform3d(featureArray, reshapedDim0, reshapedDim1, reshapedDim2);
        
        //Multiply by third row of radem.
        floatConv1dMultiplyByRadem<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(featureArray, 
                        radem + 2 * numFreqs, reshapedDim2, startPosition, numElements,
                        normConstant);
        //Last H-transform.
        floatCudaHTransform3d(featureArray, reshapedDim0, reshapedDim1, reshapedDim2);

        //Multiply by chiArr; take the sine and cosine of elements of
        //featureArray, multiply by scalingTerm, and transfer to outputArray.
        rbfConvFloatPostProcess(featureArray, chiArr,
            outputArray, reshapedDim0, reshapedDim1,
            reshapedDim2, startPosition, numFreqs,
            scalingTerm);
    }

    return "no_error";
}



//This function generates but does not sum random features for an
//input array reshapedX of input type double. The sine and cosine
//random features are stored in two arrays supplied by caller,
//who can then either sum or further modify as appropriate.
const char *doubleConvRBFFeatureGen(int8_t *radem, double *reshapedX,
                double *featureArray, double *chiArr, double *outputArray,
                int reshapedDim0, int reshapedDim1, int reshapedDim2,
                int numFreqs, double scalingTerm){

    int numElements = reshapedDim0 * reshapedDim1 * reshapedDim2;
    //This is the Hadamard normalization constant.
    float normConstant = log2(reshapedDim2) / 2;
    normConstant = 1 / pow(2, normConstant);
    int blocksPerGrid = (numElements + DEFAULT_THREADS_PER_BLOCK - 1) / 
                DEFAULT_THREADS_PER_BLOCK;

    int numRepeats = (numFreqs + reshapedDim2 - 1) / reshapedDim2;
    int i, startPosition;

    for (i=0; i < numRepeats; i++){
        startPosition = i * reshapedDim2;
    
        //Copy input into featureArray while multiplying by first row of radem.
        doubleConv1dRademAndCopy<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(reshapedX, 
                        featureArray, radem, reshapedDim2, startPosition, numElements,
                        normConstant);
        //First H-transform.
        doubleCudaHTransform3d(featureArray, reshapedDim0, reshapedDim1, reshapedDim2);

        //Multiply by second row of radem.
        doubleConv1dMultiplyByRadem<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(featureArray, 
                        radem + numFreqs, reshapedDim2, startPosition, numElements,
                        normConstant);
        //Second H-transform.
        doubleCudaHTransform3d(featureArray, reshapedDim0, reshapedDim1, reshapedDim2);
        
        //Multiply by third row of radem.
        doubleConv1dMultiplyByRadem<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(featureArray, 
                        radem + 2 * numFreqs, reshapedDim2, startPosition, numElements,
                        normConstant);
        //Last H-transform.
        doubleCudaHTransform3d(featureArray, reshapedDim0, reshapedDim1, reshapedDim2);

        //Multiply by chiArr; take the sine and cosine of elements of
        //featureArray, multiply by scalingTerm, and transfer to output.
        rbfConvDoublePostProcess(featureArray, chiArr,
            outputArray, reshapedDim0, reshapedDim1,
            reshapedDim2, startPosition, numFreqs,
            scalingTerm);
    }

    return "no_error";
}



//Carries out the final stages of feature generation
//for RBF-based kernels with float input data.
void rbfConvFloatPostProcess(float *featureArray, float *chiArr,
        double *outputArray, int reshapedDim0, int reshapedDim1,
        int reshapedDim2, int startPosition, int numFreqs,
        double scalingTerm){
    int numElements = reshapedDim0 * reshapedDim2;
    int blockRepeats = (numElements + DEFAULT_THREADS_PER_BLOCK - 1) / DEFAULT_THREADS_PER_BLOCK;
    int outputDim1 = 2 * numFreqs;
    int log2dim2 = log2(reshapedDim2);

    floatConvRBFPostProcessKernel<<<blockRepeats, DEFAULT_THREADS_PER_BLOCK>>>(featureArray, chiArr,
            outputArray, reshapedDim1, reshapedDim2,
            outputDim1, startPosition, numElements, log2dim2,
            scalingTerm);
}




//Carries out the final stages of feature generation
//for RBF-based kernels with float input data.
void rbfConvDoublePostProcess(double *featureArray, double *chiArr,
        double *outputArray, int reshapedDim0, int reshapedDim1,
        int reshapedDim2, int startPosition, int numFreqs,
        double scalingTerm){
    int numElements = reshapedDim0 * reshapedDim2;
    int blockRepeats = (numElements + DEFAULT_THREADS_PER_BLOCK - 1) / DEFAULT_THREADS_PER_BLOCK;
    int outputDim1 = 2 * numFreqs;
    int log2dim2 = log2(reshapedDim2);

    doubleConvRBFPostProcessKernel<<<blockRepeats, DEFAULT_THREADS_PER_BLOCK>>>(featureArray, chiArr,
            outputArray, reshapedDim1, reshapedDim2,
            outputDim1, startPosition, numElements, log2dim2,
            scalingTerm);
}
