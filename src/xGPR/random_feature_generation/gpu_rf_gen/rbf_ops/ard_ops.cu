/*
* Contains specialized functions for generating random features for
* ARD RBF kernels (non-convolution).
*/

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#include "../shared_constants.h"
#include "../sharedmem.h"
#include "ard_ops.h"



//Performs the first piece of the gradient calculation for ARD kernels
//only -- multiplying the input data by the precomputed weight matrix
//and summing over rows that correspond to specific lengthscales.
template <typename T>
__global__ void ardGradSetup(double *gradientArray,
        T precomputedWeights[], T inputX[], int32_t *sigmaMap,
        double *sigmaVals, double *randomFeatures,
        int dim1, int numSetupElements, int numFreqs,
        int numLengthscales){

    int i, sigmaLoc;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int precompWRow = (tid % numFreqs);
    int gradRow = tid / numFreqs;

    T outVal;

    if (tid < numSetupElements){
        T *precompWElement = precomputedWeights + precompWRow * dim1;
        T *inputXElement = inputX + gradRow * dim1;
        double *gradientElement = gradientArray + 2 * (gradRow * numFreqs + precompWRow) * numLengthscales;
        double *randomFeature = randomFeatures + 2 * (gradRow * numFreqs + precompWRow);
        double rfVal = 0;

        for (i=0; i < dim1; i++){
            sigmaLoc = sigmaMap[i];
            outVal = precompWElement[i] * inputXElement[i];
            gradientElement[sigmaLoc] += outVal;
            rfVal += sigmaVals[i] * outVal;
        }
        *randomFeature = rfVal;
    }
}





//Multiplies the gradient array by the appropriate elements of the random
//feature array when calculating the gradient for ARD kernels only.
__global__ void ardGradRFMultiply(double *gradientArray, double *randomFeats,
        int numRFElements, int numFreqs, int numLengthscales,
        double rbfNormConstant){
    int i;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int rowNum = tid / numFreqs, colNum = tid % numFreqs;
    int gradPosition = 2 * (rowNum * numFreqs + colNum) * numLengthscales;
    int rfPosition = 2 * (rowNum * numFreqs + colNum);
    double rfVal, cosVal, sinVal;
    

    if (tid < numRFElements){
        rfVal = randomFeats[rfPosition];
        cosVal = cos(rfVal) * rbfNormConstant;
        sinVal = sin(rfVal) * rbfNormConstant;
        randomFeats[rfPosition] = cosVal;
        randomFeats[rfPosition + 1] = sinVal;

        for (i=0; i < numLengthscales; i++){
            rfVal = gradientArray[gradPosition + i];
            gradientArray[gradPosition + i] = -rfVal * sinVal;
            gradientArray[gradPosition + i + numLengthscales] = rfVal * cosVal;
        }
    }
}


//This function generates the gradient and random features
//for ARD kernels only, using precomputed weights that take
//the place of the H-transforms we would otherwise need to perform.
template <typename T>
int ardCudaGrad(nb::ndarray<T, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> inputArr,
        nb::ndarray<double, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> outputArr,
        nb::ndarray<T, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> precompWeights,
        nb::ndarray<int32_t, nb::shape<-1>, nb::device::cuda, nb::c_contig> sigmaMap,
        nb::ndarray<double, nb::shape<-1>, nb::device::cuda, nb::c_contig> sigmaVals,
        nb::ndarray<double, nb::shape<-1,-1,-1>, nb::device::cuda, nb::c_contig> gradArr,
        bool fitIntercept){

    // Perform safety checks. Any exceptions thrown here are handed off to Python
    // by the Nanobind wrapper. We do not expect the user to see these because
    // the Python code will always ensure inputs are correct -- these are a failsafe
    // -- so we do not need to provide detailed exception messages here.
    int zDim0 = inputArr.shape(0);
    int zDim1 = inputArr.shape(1);

    T *inputPtr = static_cast<T*>(inputArr.data());
    T *precompWeightsPtr = static_cast<T*>(precompWeights.data());
    double *outputPtr = static_cast<double*>(outputArr.data());
    double *gradientPtr = static_cast<double*>(gradArr.data());
    int32_t *sigmaMapPtr = static_cast<int32_t*>(sigmaMap.data());
    double *sigmaValsPtr = static_cast<double*>(sigmaVals.data());

    size_t numFreqs = precompWeights.shape(0);
    double numFreqsFlt = numFreqs;
    size_t numLengthscales = gradArr.shape(2);

    if (inputArr.shape(0) == 0 || outputArr.shape(0) != inputArr.shape(0))
        throw std::runtime_error("no datapoints");
    if (gradArr.shape(0) != outputArr.shape(0) || gradArr.shape(1) != outputArr.shape(1))
        throw std::runtime_error("Wrong array sizes.");
    if (precompWeights.shape(1) != inputArr.shape(1))
        throw std::runtime_error("Wrong array sizes.");
    if (outputArr.shape(1) != 2 * precompWeights.shape(0) || sigmaMap.shape(0) != precompWeights.shape(1))
        throw std::runtime_error("Wrong array sizes.");
    if (sigmaVals.shape(0) != sigmaMap.shape(0))
        throw std::runtime_error("Wrong array sizes.");


    T rbfNormConstant;

    if (fitIntercept)
        rbfNormConstant = std::sqrt(1.0 / (numFreqsFlt - 0.5));
    else
        rbfNormConstant = std::sqrt(1.0 / numFreqsFlt);

    int numRFElements = zDim0 * numFreqs;
    int numSetupElements = zDim0 * numFreqs;
    int blocksPerGrid;


    blocksPerGrid = (numSetupElements + DEFAULT_THREADS_PER_BLOCK - 1) / DEFAULT_THREADS_PER_BLOCK;
    ardGradSetup<T><<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(gradientPtr, precompWeightsPtr,
            inputPtr, sigmaMapPtr, sigmaValsPtr, outputPtr, zDim1, numSetupElements,
            numFreqs, numLengthscales);

    blocksPerGrid = (numRFElements + DEFAULT_THREADS_PER_BLOCK - 1) / DEFAULT_THREADS_PER_BLOCK;
    ardGradRFMultiply<<<blocksPerGrid, DEFAULT_THREADS_PER_BLOCK>>>(gradientPtr, outputPtr,
                numRFElements, numFreqs, numLengthscales, rbfNormConstant);

    return 0;
}
//Explicitly instantiate so wrappers can access.
template int ardCudaGrad<double>(nb::ndarray<double, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> inputArr,
        nb::ndarray<double, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> outputArr,
        nb::ndarray<double, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> precompWeights,
        nb::ndarray<int32_t, nb::shape<-1>, nb::device::cuda, nb::c_contig> sigmaMap,
        nb::ndarray<double, nb::shape<-1>, nb::device::cuda, nb::c_contig> sigmaVals,
        nb::ndarray<double, nb::shape<-1,-1,-1>, nb::device::cuda, nb::c_contig> gradArr,
        bool fitIntercept);
template int ardCudaGrad<float>(nb::ndarray<float, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> inputArr,
        nb::ndarray<double, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> outputArr,
        nb::ndarray<float, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> precompWeights,
        nb::ndarray<int32_t, nb::shape<-1>, nb::device::cuda, nb::c_contig> sigmaMap,
        nb::ndarray<double, nb::shape<-1>, nb::device::cuda, nb::c_contig> sigmaVals,
        nb::ndarray<double, nb::shape<-1,-1,-1>, nb::device::cuda, nb::c_contig> gradArr,
        bool fitIntercept);
