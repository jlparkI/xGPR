/*
* Contains specialized functions for generating random features for
* the RBF and related kernels (non-convolution).
*/

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#include "../shared_constants.h"
#include "../sharedmem.h"
#include "rbf_ops.h"


namespace nb = nanobind;


//Generates the RBF features. This single kernel loops over 1)
//the number of repeats then inside that loop 2) the three diagonal
//matrix multiplications and fast Hadamard transforms before
//applying 3) diagonal matmul before activation function.
template <typename T>
__global__ void rbfFeatureGenKernel(const T origData[], T cArray[],
        double *outputArray, const T chiArr[], const int8_t *radem,
        int paddedBufferSize, int log2N, int numFreqs, int inputElementsPerRow,
        int nRepeats, int rademShape2, T normConstant,
        double scalingConstant){
    int stepSize = MIN(paddedBufferSize, MAX_BASE_LEVEL_TRANSFORM);

    SharedMemory<T> shared;
    T *s_data = shared.getPointer();
    int spacing, pos = threadIdx.x;
    int lo, id1, id2;
    int tempArrPos, chiArrPos = 0;
    int inputArrPos = (blockIdx.x * inputElementsPerRow);
    int outputArrPos = (blockIdx.x * numFreqs * 2);
    T y, outputVal;
    const int8_t *rademPtr = radem;

    //Run over the number of repeats required to generate the random
    //features.
    for (int rep = 0; rep < nRepeats; rep++){
        tempArrPos = (blockIdx.x << log2N);

        //Copy original data into the temporary array.
        for (int i = threadIdx.x; i < paddedBufferSize; i += blockDim.x){
            if (i < inputElementsPerRow)
                cArray[i + tempArrPos] = origData[i + inputArrPos];
            else
                cArray[i + tempArrPos] = 0;
        }

        //Run over three repeats for the SORF procedure.
        for (int sorfRep = 0; sorfRep < 3; sorfRep++){
            rademPtr = radem + paddedBufferSize * rep + sorfRep * rademShape2;
            tempArrPos = (blockIdx.x << log2N);

            for (int hStep = 0; hStep < paddedBufferSize; hStep+=stepSize){
                for (int i = threadIdx.x; i < stepSize; i += blockDim.x)
                    s_data[i] = cArray[i + tempArrPos];

                __syncthreads();

                //Multiply by the diagonal array here.
                for (int i = threadIdx.x; i < stepSize; i += blockDim.x)
                    s_data[i] = s_data[i] * rademPtr[i] * normConstant;

                rademPtr += stepSize;

                id1 = (pos << 1);
                id2 = id1 + 1;
                __syncthreads();
                y = s_data[id2];
                s_data[id2] = s_data[id1] - y;
                s_data[id1] += y;

                for (spacing = 2; spacing < stepSize; spacing <<= 1){
                    //Equivalent to pos mod spacing if spacing is a power of 2,
                    //which here is always true.
                    lo = pos & (spacing - 1);
                    id1 = ((pos - lo) << 1) + lo;
                    id2 = id1 + spacing;
                    __syncthreads();
                    y = s_data[id2];
                    s_data[id2] = s_data[id1] - y;
                    s_data[id1] += y;
                }
                __syncthreads();

                for (int i = threadIdx.x; i < stepSize; i += blockDim.x)
                    cArray[i + tempArrPos] = s_data[i];

                tempArrPos += stepSize;
                __syncthreads();
            }

            //A less efficient global memory procedure to complete the FHT
            //for long arrays.
            if (paddedBufferSize > MAX_BASE_LEVEL_TRANSFORM){
                tempArrPos = (blockIdx.x << log2N);

                for (int spacing = stepSize; spacing < paddedBufferSize; spacing <<= 1){

                    for (int k = 0; k < paddedBufferSize; k += (spacing << 1)){
                        for (int i = threadIdx.x; i < spacing; i += blockDim.x){
                            id1 = i + k + tempArrPos;
                            id2 = id1 + spacing;
                            y = cArray[id2];
                            cArray[id2] = cArray[id1] - y;
                            cArray[id1] += y;
                        }
                        __syncthreads();
                    }
                }
            }
        }
        //Now take the results stored in the temporary array, apply the
        //activation function, and populate the output array. Note that
        //we multiply by 2 in the output array position since two
        //features are generated for each frequency sampled.
        tempArrPos = (blockIdx.x << log2N);

        for (int i = threadIdx.x; i < paddedBufferSize; i += blockDim.x){
            if ((i + chiArrPos) >= numFreqs)
                break;
            outputVal = chiArr[chiArrPos + i] * cArray[tempArrPos + i];
            outputArray[outputArrPos + 2 * i] = scalingConstant * cos(outputVal);
            outputArray[outputArrPos + 2 * i + 1] = scalingConstant * sin(outputVal);
        }

        chiArrPos += paddedBufferSize;
        outputArrPos += 2 * paddedBufferSize;
        __syncthreads();

    }
}




//Generates the RBF features with gradient. This single kernel loops over 1)
//the number of repeats then inside that loop 2) the three diagonal
//matrix multiplications and fast Hadamard transforms before
//applying 3) diagonal matmul before activation function. The only difference
//from rbfFeatureGenKernel is that the gradient is also calculated.
template <typename T>
__global__ void rbfFeatureGradKernel(const T origData[], T cArray[],
        double *outputArray, const T chiArr[], const int8_t *radem,
        int paddedBufferSize, int log2N, int numFreqs, int inputElementsPerRow,
        int nRepeats, int rademShape2, T normConstant,
        double scalingConstant, double *gradient, T sigma){
    int stepSize = MIN(paddedBufferSize, MAX_BASE_LEVEL_TRANSFORM);

    SharedMemory<T> shared;
    T *s_data = shared.getPointer();
    int spacing, pos = threadIdx.x;
    int lo, id1, id2;
    int tempArrPos, chiArrPos = 0;
    int inputArrPos = (blockIdx.x * inputElementsPerRow);
    int outputArrPos = (blockIdx.x * numFreqs * 2);
    T y, outputVal;
    const int8_t *rademPtr = radem;

    //Run over the number of repeats required to generate the random
    //features.
    for (int rep = 0; rep < nRepeats; rep++){
        tempArrPos = (blockIdx.x << log2N);

        //Copy original data into the temporary array.
        for (int i = threadIdx.x; i < paddedBufferSize; i += blockDim.x){
            if (i < inputElementsPerRow)
                cArray[i + tempArrPos] = origData[i + inputArrPos];
            else
                cArray[i + tempArrPos] = 0;
        }

        //Run over three repeats for the SORF procedure.
        for (int sorfRep = 0; sorfRep < 3; sorfRep++){
            rademPtr = radem + paddedBufferSize * rep + sorfRep * rademShape2;
            tempArrPos = (blockIdx.x << log2N);

            for (int hStep = 0; hStep < paddedBufferSize; hStep+=stepSize){
                for (int i = threadIdx.x; i < stepSize; i += blockDim.x)
                    s_data[i] = cArray[i + tempArrPos];

                __syncthreads();

                //Multiply by the diagonal array here.
                for (int i = threadIdx.x; i < stepSize; i += blockDim.x)
                    s_data[i] = s_data[i] * rademPtr[i] * normConstant;

                rademPtr += stepSize;

                id1 = (pos << 1);
                id2 = id1 + 1;
                __syncthreads();
                y = s_data[id2];
                s_data[id2] = s_data[id1] - y;
                s_data[id1] += y;

                for (spacing = 2; spacing < stepSize; spacing <<= 1){
                    //Equivalent to pos mod spacing if spacing is a power of 2,
                    //which here is always true.
                    lo = pos & (spacing - 1);
                    id1 = ((pos - lo) << 1) + lo;
                    id2 = id1 + spacing;
                    __syncthreads();
                    y = s_data[id2];
                    s_data[id2] = s_data[id1] - y;
                    s_data[id1] += y;
                }
                __syncthreads();

                for (int i = threadIdx.x; i < stepSize; i += blockDim.x)
                    cArray[i + tempArrPos] = s_data[i];

                tempArrPos += stepSize;
                __syncthreads();
            }

            //A less efficient global memory procedure to complete the FHT
            //for long arrays.
            if (paddedBufferSize > MAX_BASE_LEVEL_TRANSFORM){
                tempArrPos = (blockIdx.x << log2N);

                for (int spacing = stepSize; spacing < paddedBufferSize; spacing <<= 1){

                    for (int k = 0; k < paddedBufferSize; k += (spacing << 1)){
                        for (int i = threadIdx.x; i < spacing; i += blockDim.x){
                            id1 = i + k + tempArrPos;
                            id2 = id1 + spacing;
                            y = cArray[id2];
                            cArray[id2] = cArray[id1] - y;
                            cArray[id1] += y;
                        }
                        __syncthreads();
                    }
                }
            }
        }
        //Now take the results stored in the temporary array, apply the
        //activation function, and populate the output array. Note that
        //we multiply by 2 in the output array position since two
        //features are generated for each frequency sampled.
        tempArrPos = (blockIdx.x << log2N);

        for (int i = threadIdx.x; i < paddedBufferSize; i += blockDim.x){
            if ((i + chiArrPos) >= numFreqs)
                break;
            outputVal = chiArr[chiArrPos + i] * cArray[tempArrPos + i];
            double prodVal = outputVal * sigma;
            outputArray[outputArrPos + 2 * i] = scalingConstant * cos(prodVal);
            outputArray[outputArrPos + 2 * i + 1] = scalingConstant * sin(prodVal);
            gradient[outputArrPos + 2 * i] = -scalingConstant * sin(prodVal) * outputVal;
            gradient[outputArrPos + 2 * i + 1] = scalingConstant * cos(prodVal) * outputVal;
        }

        chiArrPos += paddedBufferSize;
        outputArrPos += 2 * paddedBufferSize;
        __syncthreads();

    }
}




//This function generates random features for RBF / ARD kernels, if the
//input has already been multiplied by the appropriate lengthscale values.
template <typename T>
int RBFFeatureGen(
        nb::ndarray<const T, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> inputArr,
        nb::ndarray<double, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> outputArr,
        nb::ndarray<const int8_t, nb::shape<3,1,-1>, nb::device::cuda, nb::c_contig> radem,
        nb::ndarray<const T, nb::shape<-1>, nb::device::cuda, nb::c_contig> chiArr,
        bool fitIntercept) {

    // Perform safety checks. Any exceptions thrown here are handed off to Python
    // by the Nanobind wrapper. We do not expect the user to see these because
    // the Python code will always ensure inputs are correct -- these are a failsafe
    // -- so we do not need to provide detailed exception messages here.
    int zDim0 = inputArr.shape(0);
    int zDim1 = inputArr.shape(1);
    size_t numRffs = outputArr.shape(1);
    size_t numFreqs = chiArr.shape(0);
    double numFreqsFlt = numFreqs;

    const T *inputPtr = inputArr.data();
    double *outputPtr = outputArr.data();
    const T *chiPtr = chiArr.data();
    const int8_t *rademPtr = radem.data();

    if (inputArr.shape(0) == 0 || outputArr.shape(0) != inputArr.shape(0))
        throw std::runtime_error("no datapoints");
    if (numRffs < 2 || (numRffs & 1) != 0)
        throw std::runtime_error("last dim of output must be even number");
    if ( (2 * numFreqs) != numRffs || numFreqs > radem.shape(2) )
        throw std::runtime_error("incorrect number of rffs and or freqs.");

    double expectedNFreq = (zDim1 > 2) ? static_cast<double>(zDim1) : 2.0;
    double log2Freqs = std::log2(expectedNFreq);
    log2Freqs = std::ceil(log2Freqs);
    int paddedBufferSize = std::pow(2, log2Freqs);

    if (radem.shape(2) % paddedBufferSize != 0)
        throw std::runtime_error("incorrect number of rffs and or freqs.");

    T rbfNormConstant;

    if (fitIntercept)
        rbfNormConstant = std::sqrt(1.0 / (numFreqsFlt - 0.5));
    else
        rbfNormConstant = std::sqrt(1.0 / numFreqsFlt);

    //This is the Hadamard normalization constant.
    T normConstant = log2(paddedBufferSize) / 2;
    normConstant = 1 / pow(2, normConstant);
    int numRepeats = (numFreqs + paddedBufferSize - 1) / paddedBufferSize;
    int stepSize = MIN(MAX_BASE_LEVEL_TRANSFORM, paddedBufferSize);
    int log2N = log2(paddedBufferSize);

    T *featureArray;
    if (cudaMalloc(&featureArray, sizeof(T) * zDim0 * paddedBufferSize) != cudaSuccess) {
        cudaFree(featureArray);
        throw std::runtime_error("out of memory on cuda");
        return 1;
    };

    rbfFeatureGenKernel<T><<<zDim0, stepSize / 2, stepSize * sizeof(T)>>>(inputPtr,
            featureArray, outputPtr, chiPtr, rademPtr, paddedBufferSize, log2N, numFreqs, zDim1,
            numRepeats, radem.shape(2), normConstant, rbfNormConstant);

    cudaFree(featureArray);
    return 0;
}
//Instantiate templates so Cython / PyBind wrappers can import.
template int RBFFeatureGen<double>(
        nb::ndarray<const double, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> inputArr,
        nb::ndarray<double, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> outputArr,
        nb::ndarray<const int8_t, nb::shape<3,1,-1>, nb::device::cuda, nb::c_contig> radem,
        nb::ndarray<const double, nb::shape<-1>, nb::device::cuda, nb::c_contig> chiArr,
        bool fitIntercept);
template int RBFFeatureGen<float>(
        nb::ndarray<const float, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> inputArr,
        nb::ndarray<double, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> outputArr,
        nb::ndarray<const int8_t, nb::shape<3,1,-1>, nb::device::cuda, nb::c_contig> radem,
        nb::ndarray<const float, nb::shape<-1>, nb::device::cuda, nb::c_contig> chiArr,
        bool fitIntercept);


//This function generates random features for RBF kernels ONLY
//(NOT ARD), and simultaneously generates the gradient, storing
//it in a separate array.
template <typename T>
int RBFFeatureGrad(
        nb::ndarray<const T, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> inputArr,
        nb::ndarray<double, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> outputArr,
        nb::ndarray<double, nb::shape<-1,-1,1>, nb::device::cuda, nb::c_contig> gradArr,
        nb::ndarray<const int8_t, nb::shape<3,1,-1>, nb::device::cuda, nb::c_contig> radem,
        nb::ndarray<const T, nb::shape<-1>, nb::device::cuda, nb::c_contig> chiArr,
        float sigma, bool fitIntercept) {

    // Perform safety checks. Any exceptions thrown here are handed off to Python
    // by the Nanobind wrapper. We do not expect the user to see these because
    // the Python code will always ensure inputs are correct -- these are a failsafe
    // -- so we do not need to provide detailed exception messages here.
    int zDim0 = inputArr.shape(0);
    int zDim1 = inputArr.shape(1);
    size_t numRffs = outputArr.shape(1);
    size_t numFreqs = chiArr.shape(0);
    double numFreqsFlt = numFreqs;

    const T *inputPtr = inputArr.data();
    double *outputPtr = outputArr.data();
    double *gradientPtr = gradArr.data();
    const T *chiPtr = chiArr.data();
    const int8_t *rademPtr = radem.data();

    if (inputArr.shape(0) == 0 || outputArr.shape(0) != inputArr.shape(0))
        throw std::runtime_error("no datapoints");
    if (numRffs < 2 || (numRffs & 1) != 0)
        throw std::runtime_error("last dim of output must be even number");
    if ( (2 * numFreqs) != numRffs || numFreqs > radem.shape(2) )
        throw std::runtime_error("incorrect number of rffs and or freqs.");
    if (gradArr.shape(0) != outputArr.shape(0) || gradArr.shape(1) != outputArr.shape(1))
        throw std::runtime_error("Wrong array sizes.");

    double expectedNFreq = (zDim1 > 2) ? static_cast<double>(zDim1) : 2.0;
    double log2Freqs = std::log2(expectedNFreq);
    log2Freqs = std::ceil(log2Freqs);
    int paddedBufferSize = std::pow(2, log2Freqs);

    if (radem.shape(2) % paddedBufferSize != 0)
        throw std::runtime_error("incorrect number of rffs and or freqs.");

    T rbfNormConstant;

    if (fitIntercept)
        rbfNormConstant = std::sqrt(1.0 / (numFreqsFlt - 0.5));
    else
        rbfNormConstant = std::sqrt(1.0 / numFreqsFlt);


    //This is the Hadamard normalization constant.
    T normConstant = log2(paddedBufferSize) / 2;
    normConstant = 1 / pow(2, normConstant);
    int numRepeats = (numFreqs + paddedBufferSize - 1) / paddedBufferSize;
    int stepSize = MIN(MAX_BASE_LEVEL_TRANSFORM, paddedBufferSize);
    int log2N = log2(paddedBufferSize);

    T *featureArray;
    if (cudaMalloc(&featureArray, sizeof(T) * zDim0 * paddedBufferSize) != cudaSuccess) {
        cudaFree(featureArray);
        throw std::runtime_error("out of memory on cuda");
        return 1;
    };

    rbfFeatureGradKernel<T><<<zDim0, stepSize / 2, stepSize * sizeof(T)>>>(inputPtr,
            featureArray, outputPtr, chiPtr, rademPtr, paddedBufferSize, log2N, numFreqs, zDim1,
            numRepeats, radem.shape(2), normConstant, rbfNormConstant, gradientPtr,
            sigma);

    cudaFree(featureArray);
    return 0;
}
//Instantiate templates so Cython / PyBind wrappers can import.
template int RBFFeatureGrad<double>(
        nb::ndarray<const double, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> inputArr,
        nb::ndarray<double, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> outputArr,
        nb::ndarray<double, nb::shape<-1,-1,1>, nb::device::cuda, nb::c_contig> gradArr,
        nb::ndarray<const int8_t, nb::shape<3,1,-1>, nb::device::cuda, nb::c_contig> radem,
        nb::ndarray<const double, nb::shape<-1>, nb::device::cuda, nb::c_contig> chiArr,
        float sigma, bool fitIntercept);
template int RBFFeatureGrad<float>(
        nb::ndarray<const float, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> inputArr,
        nb::ndarray<double, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> outputArr,
        nb::ndarray<double, nb::shape<-1,-1,1>, nb::device::cuda, nb::c_contig> gradArr,
        nb::ndarray<const int8_t, nb::shape<3,1,-1>, nb::device::cuda, nb::c_contig> radem,
        nb::ndarray<const float, nb::shape<-1>, nb::device::cuda, nb::c_contig> chiArr,
        float sigma, bool fitIntercept);
