/*
* Contains routines needed specifically for generating features for RBF-based
* convolution kernels (FHTConv1d, GraphConv) and calculating their gradients.
*/
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../float_array_operations.h"
#include "../double_array_operations.h"
#include "rbf_convolution.h"

#define DEFAULT_THREADS_PER_BLOCK 256
#define DEFAULT_THREADS_PER_BLREDUCE 32




//Performs an elementwise multiplication of a row of a [3,1,P x S] array against the
//float [N,M,S] input array. Note that the dimensions must be checked before calling
//-- done by the wrapper -- and that only S elements of the appropriate row of
//the [3, 1, P x S] array are used.
__global__ void floatConv1dRBFRademMultiply(float *cArray, int8_t *rademArray,
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
__global__ void doubleConv1dRBFRademMultiply(double *cArray, int8_t *rademArray,
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
__global__ void floatConv1dRBFRademAndCopy(float *inputArray, float *featureArray,
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
__global__ void doubleConv1dRBFRademAndCopy(double *inputArray, double *featureArray,
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
    //We multiply startPosition by 2 here because outputArray must have
    //a block of cosine features followed by a block of sine features etc.
    int outputLoc = row * outputDim1 + column + 2 * startPosition;
    float *chiVal = chiArr + startPosition + column;
    float chiProd, sinSum = 0, cosSum = 0;

    if (tid < numElements){
        for (i=0; i < dim1; i++){
            chiProd = *chiVal * featureArray[inputLoc];
            cosSum += cosf(chiProd);
            sinSum += sinf(chiProd);
            inputLoc += dim2;
        }
        outputArray[outputLoc] = cosSum * scalingTerm;
        outputArray[outputLoc + dim2] = sinSum * scalingTerm;
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
    //We multiply startPosition by 2 here because outputArray must have
    //a block of cosine features followed by a block of sine features etc.
    int outputLoc = row * outputDim1 + column + 2 * startPosition;
    double *chiVal = chiArr + startPosition + column;
    double chiProd, sinSum = 0, cosSum = 0;

    if (tid < numElements){
        for (i=0; i < dim1; i++){
            chiProd = *chiVal * featureArray[inputLoc];
            cosSum += cos(chiProd);
            sinSum += sin(chiProd);
            inputLoc += dim2;
        }
        outputArray[outputLoc] = cosSum * scalingTerm;
        outputArray[outputLoc + dim2] = sinSum * scalingTerm;
    }
}




//Performs the final steps in feature AND gradient generation for RBF-based convolution
//kernels -- multiplying by chiArr, taking sine or cosine and adding
//to the appropriate elements of outputArray, then adding the same to the 
//appropriate elements of gradientArray.
__global__ void floatConvRBFGradProcessKernel(float *featureArray, float *chiArr,
            double *outputArray, int dim1, int dim2,
            int outputDim1, int startPosition, int numElements,
            int log2dim2, double scalingTerm,
            double sigma, double *gradientArray)
{
    int i;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int column = (tid & (dim2 - 1));
    int row = (tid >> log2dim2);
    int inputLoc = row * dim1 * dim2 + column;
    //We multiply startPosition by 2 here because outputArray must have
    //a block of cosine features followed by a block of sine features etc.
    int outputLoc = row * outputDim1 + column + 2 * startPosition;
    float *chiVal = chiArr + startPosition + column;
    float chiProd, sinSum = 0, cosSum = 0, sinVal, cosVal;
    float gradSinVal = 0, gradCosVal = 0;

    if (tid < numElements){
        for (i=0; i < dim1; i++){
            chiProd = *chiVal * featureArray[inputLoc];
            cosVal = cosf(chiProd * sigma);
            sinVal = sinf(chiProd * sigma);

            cosSum += cosVal;
            sinSum += sinVal;
            //These are the derivatives.
            gradCosVal -= sinVal * chiProd;
            gradSinVal += cosVal * chiProd;
            inputLoc += dim2;
        }
        outputArray[outputLoc] = cosSum * scalingTerm;
        outputArray[outputLoc + dim2] = sinSum * scalingTerm;

        gradientArray[outputLoc] = gradCosVal * scalingTerm;
        gradientArray[outputLoc + dim2] = gradSinVal * scalingTerm;
    }
}



//Performs the final steps in feature AND gradient generation for RBF-based convolution
//kernels -- multiplying by chiArr, taking sine or cosine and adding
//to the appropriate elements of outputArray, then adding the same to the 
//appropriate elements of gradientArray.
__global__ void doubleConvRBFGradProcessKernel(double *featureArray, double *chiArr,
            double *outputArray, int dim1, int dim2,
            int outputDim1, int startPosition, int numElements,
            int log2dim2, double scalingTerm,
            double sigma, double *gradientArray)
{
    int i;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int column = (tid & (dim2 - 1));
    int row = (tid >> log2dim2);
    int inputLoc = row * dim1 * dim2 + column;
    //We multiply startPosition by 2 here because outputArray must have
    //a block of cosine features followed by a block of sine features etc.
    int outputLoc = row * outputDim1 + column + 2 * startPosition;
    double *chiVal = chiArr + startPosition + column;
    double chiProd, sinSum = 0, cosSum = 0, sinVal, cosVal;
    double gradSinVal = 0, gradCosVal = 0;

    if (tid < numElements){
        for (i=0; i < dim1; i++){
            chiProd = *chiVal * featureArray[inputLoc];
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
        outputArray[outputLoc + dim2] = sinSum * scalingTerm;

        gradientArray[outputLoc] = gradCosVal * scalingTerm;
        gradientArray[outputLoc + dim2] = gradSinVal * scalingTerm;
    }
}



//Carries out the final stages of feature generation
//for RBF-based kernels with float input data by wrapping
//the PostProcess kernel.
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
//for RBF-based kernels with double input data by wrapping
//the PostProcess kernel.
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



//Carries out the final stages of feature generation
//for RBF-based kernels with float input data WHILE
//generating the gradient information, by wrapping
//the GradProcess kernel.
void rbfConvFloatGradProcess(float *featureArray, float *chiArr,
        double *outputArray, int reshapedDim0, int reshapedDim1,
        int reshapedDim2, int startPosition, int numFreqs,
        double scalingTerm, double sigma, double *gradientArray){
    int numElements = reshapedDim0 * reshapedDim2;
    int blockRepeats = (numElements + DEFAULT_THREADS_PER_BLOCK - 1) / DEFAULT_THREADS_PER_BLOCK;
    int outputDim1 = 2 * numFreqs;
    int log2dim2 = log2(reshapedDim2);

    floatConvRBFGradProcessKernel<<<blockRepeats, DEFAULT_THREADS_PER_BLOCK>>>(featureArray, chiArr,
            outputArray, reshapedDim1, reshapedDim2,
            outputDim1, startPosition, numElements, log2dim2,
            scalingTerm, sigma, gradientArray);
}



//Carries out the final stages of feature generation
//for RBF-based kernels with double input data WHILE
//generating the gradient information, by wrapping
//the GradProcess kernel.
void rbfConvDoubleGradProcess(double *featureArray, double *chiArr,
        double *outputArray, int reshapedDim0, int reshapedDim1,
        int reshapedDim2, int startPosition, int numFreqs,
        double scalingTerm, double sigma, double *gradientArray){
    int numElements = reshapedDim0 * reshapedDim2;
    int blockRepeats = (numElements + DEFAULT_THREADS_PER_BLOCK - 1) / DEFAULT_THREADS_PER_BLOCK;
    int outputDim1 = 2 * numFreqs;
    int log2dim2 = log2(reshapedDim2);

    doubleConvRBFGradProcessKernel<<<blockRepeats, DEFAULT_THREADS_PER_BLOCK>>>(featureArray, chiArr,
            outputArray, reshapedDim1, reshapedDim2,
            outputDim1, startPosition, numElements, log2dim2,
            scalingTerm, sigma, gradientArray);
}




//This function generates and sums random features for an
//input array reshapedX of input type float.
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
        floatConv1dRBFRademAndCopy<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(reshapedX, 
                        featureArray, radem, reshapedDim2, startPosition, numElements,
                        normConstant);
        //First H-transform.
        floatCudaHTransform3d(featureArray, reshapedDim0, reshapedDim1, reshapedDim2);

        //Multiply by second row of radem.
        floatConv1dRBFRademMultiply<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(featureArray, 
                        radem + numFreqs, reshapedDim2, startPosition, numElements,
                        normConstant);
        //Second H-transform.
        floatCudaHTransform3d(featureArray, reshapedDim0, reshapedDim1, reshapedDim2);
        
        //Multiply by third row of radem.
        floatConv1dRBFRademMultiply<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(featureArray, 
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



//This function generates and sums random features for an
//input array reshapedX of input type double.
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
        doubleConv1dRBFRademAndCopy<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(reshapedX, 
                        featureArray, radem, reshapedDim2, startPosition, numElements,
                        normConstant);
        //First H-transform.
        doubleCudaHTransform3d(featureArray, reshapedDim0, reshapedDim1, reshapedDim2);

        //Multiply by second row of radem.
        doubleConv1dRBFRademMultiply<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(featureArray, 
                        radem + numFreqs, reshapedDim2, startPosition, numElements,
                        normConstant);
        //Second H-transform.
        doubleCudaHTransform3d(featureArray, reshapedDim0, reshapedDim1, reshapedDim2);
        
        //Multiply by third row of radem.
        doubleConv1dRBFRademMultiply<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(featureArray, 
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




//This function generates and sums random features for an
//input array reshapedX of input type float WHILE also
//generating gradient information and storing this in
//a separate array. This gradient is only applicable
//in cases where all of the features share the same
//lengthscale; ARD-type kernels require a more complicated
//gradient calculation not implemented here.
const char *floatConvRBFFeatureGrad(int8_t *radem, float *reshapedX,
            float *featureArray, float *chiArr, double *outputArray,     
            double *gradientArray, double sigma,
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
        floatConv1dRBFRademAndCopy<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(reshapedX, 
                        featureArray, radem, reshapedDim2, startPosition, numElements,
                        normConstant);
        //First H-transform.
        floatCudaHTransform3d(featureArray, reshapedDim0, reshapedDim1, reshapedDim2);

        //Multiply by second row of radem.
        floatConv1dRBFRademMultiply<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(featureArray, 
                        radem + numFreqs, reshapedDim2, startPosition, numElements,
                        normConstant);
        //Second H-transform.
        floatCudaHTransform3d(featureArray, reshapedDim0, reshapedDim1, reshapedDim2);
        
        //Multiply by third row of radem.
        floatConv1dRBFRademMultiply<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(featureArray, 
                        radem + 2 * numFreqs, reshapedDim2, startPosition, numElements,
                        normConstant);
        //Last H-transform.
        floatCudaHTransform3d(featureArray, reshapedDim0, reshapedDim1, reshapedDim2);

        //Multiply by chiArr; take the sine and cosine of elements of
        //featureArray, multiply by scalingTerm, transfer to output
        //AND at the same time calculate the gradient terms, using
        //them to populate gradientArray.
        rbfConvFloatGradProcess(featureArray, chiArr,
            outputArray, reshapedDim0, reshapedDim1,
            reshapedDim2, startPosition, numFreqs,
            scalingTerm, sigma, gradientArray);
    }

    return "no_error";
}



//This function generates and sums random features for an
//input array reshapedX of input type double WHILE also
//generating gradient information and storing this in
//a separate array. This gradient is only applicable
//in cases where all of the features share the same
//lengthscale; ARD-type kernels require a more complicated
//gradient calculation not implemented here.
const char *doubleConvRBFFeatureGrad(int8_t *radem, double *reshapedX,
                double *featureArray, double *chiArr, double *outputArray,
                double *gradientArray, double sigma,
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
        doubleConv1dRBFRademAndCopy<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(reshapedX, 
                        featureArray, radem, reshapedDim2, startPosition, numElements,
                        normConstant);
        //First H-transform.
        doubleCudaHTransform3d(featureArray, reshapedDim0, reshapedDim1, reshapedDim2);

        //Multiply by second row of radem.
        doubleConv1dRBFRademMultiply<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(featureArray, 
                        radem + numFreqs, reshapedDim2, startPosition, numElements,
                        normConstant);
        //Second H-transform.
        doubleCudaHTransform3d(featureArray, reshapedDim0, reshapedDim1, reshapedDim2);
        
        //Multiply by third row of radem.
        doubleConv1dRBFRademMultiply<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(featureArray, 
                        radem + 2 * numFreqs, reshapedDim2, startPosition, numElements,
                        normConstant);
        //Last H-transform.
        doubleCudaHTransform3d(featureArray, reshapedDim0, reshapedDim1, reshapedDim2);

        //Multiply by chiArr; take the sine and cosine of elements of
        //featureArray, multiply by scalingTerm, transfer to output
        //AND at the same time calculate the gradient terms, using
        //them to populate gradientArray.
        rbfConvDoubleGradProcess(featureArray, chiArr,
            outputArray, reshapedDim0, reshapedDim1,
            reshapedDim2, startPosition, numFreqs,
            scalingTerm, sigma, gradientArray);
    }

    return "no_error";
}
