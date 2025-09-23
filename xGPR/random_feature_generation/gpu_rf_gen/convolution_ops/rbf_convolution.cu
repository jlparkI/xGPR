/*
* Contains routines needed specifically for generating features for RBF-based
* convolution kernels (FHTConv1d, GraphConv) and calculating their gradients.
*/
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#include "../shared_constants.h"
#include "../sharedmem.h"
#include "rbf_convolution.h"

//Generates the Conv kernel RBF features. This single kernel loops over 1) kmers
//then 2) the number of repeats then inside that loop 3) the three diagonal
//matrix multiplications and fast Hadamard transforms before
//applying 4) diagonal matmul before activation function.
template <typename T>
__global__ void convRBFFeatureGenKernel(const T origData[], T cArray[],
        double *outputArray, const T chiArr[], const int8_t *radem,
        int paddedBufferSize, int log2N, int numFreqs, int xDim1, int xDim2,
        int nRepeats, int rademShape2, T normConstant,
        double scalingConstant, int scalingType,
        int convWidth, const int32_t *seqlengths){

    int stepSize = MIN(paddedBufferSize, MAX_BASE_LEVEL_TRANSFORM);
    int colCutoff = seqlengths[blockIdx.x] - convWidth + 1;

    SharedMemory<T> shared;
    T *s_data = shared.getPointer();
    int spacing, pos = threadIdx.x;
    int lo, id1, id2;
    int tempArrPos, chiArrPos = 0, inputCutoff = xDim2 * convWidth;
    int inputArrPos = (blockIdx.x * xDim1 * xDim2);
    int outputArrPos = (blockIdx.x * numFreqs * 2);
    T y, outputVal, modifiedScaling = scalingConstant;

    const int8_t *rademPtr = radem;

    switch (scalingType){
        case 0:
            break;
        case 1:
            modifiedScaling = modifiedScaling / sqrt( (double) colCutoff);
            break;
        case 2:
            modifiedScaling = modifiedScaling / (double) colCutoff;
            break;
    }

    //Loop over the kmers in this stretch.
    for (int kmer = 0; kmer < colCutoff; kmer++){
        chiArrPos = 0;
        outputArrPos = (blockIdx.x * numFreqs * 2);
        inputArrPos = (blockIdx.x * xDim1 * xDim2) + kmer * xDim2;

        //Run over the number of repeats required to generate the random
        //features.
        for (int rep = 0; rep < nRepeats; rep++){
            tempArrPos = (blockIdx.x << log2N);

            //Copy original data into the temporary array.
            for (int i = threadIdx.x; i < paddedBufferSize; i += blockDim.x){
                if (i < inputCutoff)
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
                outputArray[outputArrPos + 2 * i] += modifiedScaling * cos(outputVal);
                outputArray[outputArrPos + 2 * i + 1] += modifiedScaling * sin(outputVal);
            }

            chiArrPos += paddedBufferSize;
            outputArrPos += 2 * paddedBufferSize;
            __syncthreads();
        }
    }
}




//Generates the Conv kernel RBF features together with the gradient. This single
//kernel loops over 1) kmers then 2) the number of repeats then inside that
//loop 3) the three diagonal matrix multiplications and fast Hadamard transforms
//before applying 4) diagonal matmul before activation function.
template <typename T>
__global__ void convRBFFeatureGradKernel(const T origData[], T cArray[],
        double *outputArray, const T chiArr[], const int8_t *radem,
        int paddedBufferSize, int log2N, int numFreqs, int xDim1, int xDim2,
        int nRepeats, int rademShape2, T normConstant,
        double scalingConstant, int scalingType,
        int convWidth, const int32_t *seqlengths,
        double *gradient, T sigma){

    int stepSize = MIN(paddedBufferSize, MAX_BASE_LEVEL_TRANSFORM);
    int colCutoff = seqlengths[blockIdx.x] - convWidth + 1;

    SharedMemory<T> shared;
    T *s_data = shared.getPointer();
    int spacing, pos = threadIdx.x;
    int lo, id1, id2;
    int tempArrPos, chiArrPos = 0, inputCutoff = xDim2 * convWidth;
    int inputArrPos = (blockIdx.x * xDim1 * xDim2);
    int outputArrPos = (blockIdx.x * numFreqs * 2);
    T y, outputVal, modifiedScaling = scalingConstant;

    const int8_t *rademPtr = radem;

    switch (scalingType){
        case 0:
            break;
        case 1:
            modifiedScaling = modifiedScaling / sqrt( (double) colCutoff);
            break;
        case 2:
            modifiedScaling = modifiedScaling / (double) colCutoff;
            break;
    }

    //Loop over the kmers in this stretch.
    for (int kmer = 0; kmer < colCutoff; kmer++){
        chiArrPos = 0;
        outputArrPos = (blockIdx.x * numFreqs * 2);
        inputArrPos = (blockIdx.x * xDim1 * xDim2) + kmer * xDim2;

        //Run over the number of repeats required to generate the random
        //features.
        for (int rep = 0; rep < nRepeats; rep++){
            tempArrPos = (blockIdx.x << log2N);

            //Copy original data into the temporary array.
            for (int i = threadIdx.x; i < paddedBufferSize; i += blockDim.x){
                if (i < inputCutoff)
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
                outputArray[outputArrPos + 2 * i] += modifiedScaling * cos(prodVal);
                outputArray[outputArrPos + 2 * i + 1] += modifiedScaling * sin(prodVal);
                gradient[outputArrPos + 2 * i] -= modifiedScaling * sin(prodVal) * outputVal;
                gradient[outputArrPos + 2 * i + 1] += modifiedScaling * cos(prodVal) * outputVal;
            }

            chiArrPos += paddedBufferSize;
            outputArrPos += 2 * paddedBufferSize;
            __syncthreads();
        }
    }
}






//This function generates and sums random features for a Conv1d RBF-type kernel.
template <typename T>
int convRBFFeatureGen(nb::ndarray<T, nb::shape<-1,-1,-1>, nb::device::cuda, nb::c_contig> inputArr,
        nb::ndarray<double, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> outputArr,
        nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cuda, nb::c_contig> radem,
        nb::ndarray<T, nb::shape<-1>, nb::device::cuda, nb::c_contig> chiArr,
        nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> seqlengths,
        int convWidth, int scalingType) {

    // Perform safety checks. Any exceptions thrown here are handed off to Python
    // by the Nanobind wrapper. We do not expect the user to see these because
    // the Python code will always ensure inputs are correct -- these are a failsafe
    // -- so we do not need to provide detailed exception messages here.
    int zDim0 = inputArr.shape(0);
    int zDim1 = inputArr.shape(1);
    int zDim2 = inputArr.shape(2);
    size_t numRffs = outputArr.shape(1);
    size_t numFreqs = chiArr.shape(0);
    double scalingTerm = std::sqrt(1.0 / static_cast<double>(numFreqs));

    T *inputPtr = inputArr.data();
    double *outputPtr = outputArr.data();
    T *chiPtr = chiArr.data();
    int8_t *rademPtr = radem.data();
    int32_t *seqlengthsPtr = seqlengths.data();

    if (inputArr.shape(0) == 0 || outputArr.shape(0) != inputArr.shape(0))
        throw std::runtime_error("no datapoints");
    if (numRffs < 2 || (numRffs & 1) != 0)
        throw std::runtime_error("last dim of output must be even number");
    if ( (2 * numFreqs) != numRffs || numFreqs > radem.shape(2) )
        throw std::runtime_error("incorrect number of rffs and or freqs.");

    if (seqlengths.shape(0) != inputArr.shape(0))
        throw std::runtime_error("wrong array sizes");
    if (static_cast<int>(inputArr.shape(1)) < convWidth || convWidth <= 0)
        throw std::runtime_error("invalid conv_width");

    double expectedNFreq = static_cast<double>(convWidth * inputArr.shape(2));
    expectedNFreq = MAX(expectedNFreq, 2);
    double log2Freqs = std::log2(expectedNFreq);
    log2Freqs = std::ceil(log2Freqs);
    int paddedBufferSize = std::pow(2, log2Freqs);

    if (radem.shape(2) % paddedBufferSize != 0)
        throw std::runtime_error("incorrect number of rffs and or freqs.");

    int32_t minSeqLength = 2147483647, maxSeqLength = 0;
    for (size_t i=0; i < seqlengths.shape(0); i++){
        if (seqlengths(i) > maxSeqLength)
            maxSeqLength = seqlengths(i);
        if (seqlengths(i) < minSeqLength)
            minSeqLength = seqlengths(i);
    }

    if (maxSeqLength > static_cast<int32_t>(inputArr.shape(1)) || minSeqLength < convWidth){
        throw std::runtime_error("All sequence lengths must be >= conv width and < "
                "array size.");
    }

    int32_t *slenCudaPtr;
    if (cudaMalloc(&slenCudaPtr, sizeof(int32_t) * seqlengths.shape(0)) != cudaSuccess) {
        cudaFree(slenCudaPtr);
        throw std::runtime_error("Cuda is out of memory");
        return 1;
    };
    if (cudaMemcpy(slenCudaPtr, seqlengthsPtr, sizeof(int32_t) * seqlengths.shape(0),
                cudaMemcpyHostToDevice) != cudaSuccess){
        cudaFree(slenCudaPtr);
        throw std::runtime_error("Cuda is out of memory");
        return 1;
    }


    //This is the Hadamard normalization constant.
    T normConstant = log2(paddedBufferSize) / 2;
    normConstant = 1 / pow(2, normConstant);

    int numRepeats = (numFreqs + paddedBufferSize - 1) / paddedBufferSize;
    int stepSize = MIN(MAX_BASE_LEVEL_TRANSFORM, paddedBufferSize);
    int log2N = log2(paddedBufferSize);

    T *featureArray;
    if (cudaMalloc(&featureArray, sizeof(T) * zDim0 * paddedBufferSize) != cudaSuccess) {
        cudaFree(slenCudaPtr);
        cudaFree(featureArray);
        throw std::runtime_error("Cuda is out of memory");
        return 1;
    };

    convRBFFeatureGenKernel<T><<<zDim0, stepSize / 2, stepSize * sizeof(T)>>>(inputPtr,
            featureArray, outputPtr, chiPtr, rademPtr, paddedBufferSize, log2N, numFreqs, zDim1, zDim2,
            numRepeats, radem.shape(2), normConstant, scalingTerm, scalingType, convWidth,
            slenCudaPtr);

    cudaFree(slenCudaPtr);
    cudaFree(featureArray);
    return 0;
}
//Explicitly instantiate so wrapper can use.
template int convRBFFeatureGen<double>(nb::ndarray<double, nb::shape<-1,-1,-1>, nb::device::cuda, nb::c_contig> inputArr,
        nb::ndarray<double, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> outputArr,
        nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cuda, nb::c_contig> radem,
        nb::ndarray<double, nb::shape<-1>, nb::device::cuda, nb::c_contig> chiArr,
        nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> seqlengths,
        int convWidth, int scalingType);
template int convRBFFeatureGen<float>(nb::ndarray<float, nb::shape<-1,-1,-1>, nb::device::cuda, nb::c_contig> inputArr,
        nb::ndarray<double, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> outputArr,
        nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cuda, nb::c_contig> radem,
        nb::ndarray<float, nb::shape<-1>, nb::device::cuda, nb::c_contig> chiArr,
        nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> seqlengths,
        int convWidth, int scalingType);



//This function generates and sums random features for an
//input array reshapedX of input type float WHILE also
//generating gradient information and storing this in
//a separate array. This gradient is only applicable
//in cases where all of the features share the same
//lengthscale; ARD-type kernels require a more complicated
//gradient calculation not implemented here.
template <typename T>
int convRBFFeatureGrad(nb::ndarray<T, nb::shape<-1,-1,-1>, nb::device::cuda, nb::c_contig> inputArr,
        nb::ndarray<double, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> outputArr,
        nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cuda, nb::c_contig> radem,
        nb::ndarray<T, nb::shape<-1>, nb::device::cuda, nb::c_contig> chiArr,
        nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> seqlengths,
        nb::ndarray<double, nb::shape<-1,-1,1>, nb::device::cuda, nb::c_contig> gradArr,
        double sigma, int convWidth, int scalingType) {

    // Perform safety checks. Any exceptions thrown here are handed off to Python
    // by the Nanobind wrapper. We do not expect the user to see these because
    // the Python code will always ensure inputs are correct -- these are a failsafe
    // -- so we do not need to provide detailed exception messages here.
    int zDim0 = inputArr.shape(0);
    int zDim1 = inputArr.shape(1);
    int zDim2 = inputArr.shape(2);
    size_t numRffs = outputArr.shape(1);
    size_t numFreqs = chiArr.shape(0);
    double scalingTerm = std::sqrt(1.0 / static_cast<double>(numFreqs));

    T *inputPtr = inputArr.data();
    double *outputPtr = outputArr.data();
    T *chiPtr = chiArr.data();
    int8_t *rademPtr = radem.data();
    int32_t *seqlengthsPtr = seqlengths.data();
    double *gradientPtr = gradArr.data();

    if (inputArr.shape(0) == 0 || outputArr.shape(0) != inputArr.shape(0))
        throw std::runtime_error("no datapoints");
    if (numRffs < 2 || (numRffs & 1) != 0)
        throw std::runtime_error("last dim of output must be even number");
    if ( (2 * numFreqs) != numRffs || numFreqs > radem.shape(2) )
        throw std::runtime_error("incorrect number of rffs and or freqs.");

    if (seqlengths.shape(0) != inputArr.shape(0))
        throw std::runtime_error("wrong array sizes");
    if (static_cast<int>(inputArr.shape(1)) < convWidth || convWidth <= 0)
        throw std::runtime_error("invalid conv_width");

    if (gradArr.shape(0) != outputArr.shape(0) || gradArr.shape(1) != outputArr.shape(1))
        throw std::runtime_error("wrong array sizes");

    double expectedNFreq = static_cast<double>(convWidth * inputArr.shape(2));
    expectedNFreq = MAX(expectedNFreq, 2);
    double log2Freqs = std::log2(expectedNFreq);
    log2Freqs = std::ceil(log2Freqs);
    int paddedBufferSize = std::pow(2, log2Freqs);

    if (radem.shape(2) % paddedBufferSize != 0)
        throw std::runtime_error("incorrect number of rffs and or freqs.");

    int32_t minSeqLength = 2147483647, maxSeqLength = 0;
    for (size_t i=0; i < seqlengths.shape(0); i++){
        if (seqlengths(i) > maxSeqLength)
            maxSeqLength = seqlengths(i);
        if (seqlengths(i) < minSeqLength)
            minSeqLength = seqlengths(i);
    }

    if (maxSeqLength > static_cast<int32_t>(inputArr.shape(1)) || minSeqLength < convWidth){
        throw std::runtime_error("All sequence lengths must be >= conv width and < "
                "array size.");
    }

    int32_t *slenCudaPtr;
    if (cudaMalloc(&slenCudaPtr, sizeof(int32_t) * seqlengths.shape(0)) != cudaSuccess) {
        cudaFree(slenCudaPtr);
        throw std::runtime_error("Cuda is out of memory");
        return 1;
    };
    if (cudaMemcpy(slenCudaPtr, seqlengthsPtr, sizeof(int32_t) * seqlengths.shape(0),
                cudaMemcpyHostToDevice) != cudaSuccess){
        cudaFree(slenCudaPtr);
        throw std::runtime_error("Cuda is out of memory");
        return 1;
    }




    //This is the Hadamard normalization constant.
    T normConstant = log2(paddedBufferSize) / 2;
    normConstant = 1 / pow(2, normConstant);

    int numRepeats = (numFreqs + paddedBufferSize - 1) / paddedBufferSize;
    int stepSize = MIN(MAX_BASE_LEVEL_TRANSFORM, paddedBufferSize);
    int log2N = log2(paddedBufferSize);

    T *featureArray;
    if (cudaMalloc(&featureArray, sizeof(T) * zDim0 * paddedBufferSize) != cudaSuccess) {
        cudaFree(slenCudaPtr);
        cudaFree(featureArray);
        throw std::runtime_error("Cuda is out of memory");
        return 1;
    };

    convRBFFeatureGradKernel<T><<<zDim0, stepSize / 2, stepSize * sizeof(T)>>>(inputPtr,
            featureArray, outputPtr, chiPtr, rademPtr, paddedBufferSize, log2N, numFreqs, zDim1, zDim2,
            numRepeats, radem.shape(2), normConstant, scalingTerm, scalingType, convWidth,
            slenCudaPtr, gradientPtr, sigma);

    cudaFree(slenCudaPtr);
    cudaFree(featureArray);
    return 0;
}
//Explicitly instantiate so wrapper can use.
template int convRBFFeatureGrad<double>(nb::ndarray<double, nb::shape<-1,-1,-1>, nb::device::cuda, nb::c_contig> inputArr,
        nb::ndarray<double, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> outputArr,
        nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cuda, nb::c_contig> radem,
        nb::ndarray<double, nb::shape<-1>, nb::device::cuda, nb::c_contig> chiArr,
        nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> seqlengths,
        nb::ndarray<double, nb::shape<-1,-1,1>, nb::device::cuda, nb::c_contig> gradArr,
        double sigma, int convWidth, int scalingType);
template int convRBFFeatureGrad<float>(nb::ndarray<float, nb::shape<-1,-1,-1>, nb::device::cuda, nb::c_contig> inputArr,
        nb::ndarray<double, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> outputArr,
        nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cuda, nb::c_contig> radem,
        nb::ndarray<float, nb::shape<-1>, nb::device::cuda, nb::c_contig> chiArr,
        nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> seqlengths,
        nb::ndarray<double, nb::shape<-1,-1,1>, nb::device::cuda, nb::c_contig> gradArr,
        double sigma, int convWidth, int scalingType);
