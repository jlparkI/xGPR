/*
* Contains all functions needed to generate features for the FastConv operation
* and other non-RBF convolution operations.
*/
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#include "../shared_constants.h"
#include "../sharedmem.h"
#include "convolution.h"

//Generates the FastConv kernel features. This single kernel loops over 1) kmers
//then 2) the number of repeats then inside that loop 3) the three diagonal
//matrix multiplications and fast Hadamard transforms before
//applying 4) the simplex projection and 5) diagonal matmul before
//activation function.
template <typename T>
__global__ void convMaxpoolFeatureGenKernel(const T origData[], T cArray[],
        double *outputArray, const T chiArr[], const int8_t *radem,
        int paddedBufferSize, int log2N, int numFreqs, int xDim1, int xDim2,
        int nRepeats, T normConstant, int convWidth,
        const int32_t *seqlengths, int rademShape2){

    int stepSize = MIN(paddedBufferSize, MAX_BASE_LEVEL_TRANSFORM);
    int colCutoff = seqlengths[blockIdx.x] - convWidth + 1;

    SharedMemory<T> shared;
    T *s_data = shared.getPointer();
    int spacing, pos = threadIdx.x;
    int lo, id1, id2;
    int tempArrPos, chiArrPos = 0, inputCutoff = xDim2 * convWidth;
    int inputArrPos = (blockIdx.x * xDim1 * xDim2);
    int outputArrPos = (blockIdx.x * numFreqs);
    T y, bufferSum, simplexProjPrefactor;
    double outputVal;

    const int8_t *rademPtr = radem;

    //Loop over the kmers in this stretch.
    for (int kmer = 0; kmer < colCutoff; kmer++){
        chiArrPos = 0;
        outputArrPos = (blockIdx.x * numFreqs);
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
            //Now apply the simplex projection to the temporary array. We
            //first have to sum the elements of the temporary array and
            //use the existing shared memory as storage to help with this.
            s_data[threadIdx.x] = 0;
            tempArrPos = (blockIdx.x << log2N);
            simplexProjPrefactor = sqrt( (T)paddedBufferSize - 1.);

            for (int i = threadIdx.x; i < (paddedBufferSize - 1); i += blockDim.x)
                s_data[threadIdx.x] += cArray[i + tempArrPos];

            __syncthreads();
            for (int i = blockDim.x/2; i > 0; i >>=1){
                if (threadIdx.x < i)
                    s_data[threadIdx.x] += s_data[threadIdx.x + i];
                __syncthreads();
            }

            if (threadIdx.x == 0)
                cArray[tempArrPos + paddedBufferSize - 1] = s_data[0] / simplexProjPrefactor;

            __syncthreads();
            bufferSum = s_data[0] / simplexProjPrefactor;
            bufferSum *= ( (sqrt( (T)paddedBufferSize) + 1) / ((T)paddedBufferSize - 1.) );
            simplexProjPrefactor = sqrt( (T)paddedBufferSize / ((T)paddedBufferSize - 1.) );

            for (int i=threadIdx.x; i < (paddedBufferSize - 1); i+=blockDim.x)
                cArray[i + tempArrPos] = (cArray[i + tempArrPos] * simplexProjPrefactor - bufferSum);

            __syncthreads();

            //Now take the results stored in the temporary array, apply the
            //activation function, and populate the output array. Note that
            //we multiply by 2 in the output array position since two
            //features are generated for each frequency sampled.
            tempArrPos = (blockIdx.x << log2N);

            for (int i = threadIdx.x; i < paddedBufferSize; i += blockDim.x){
                if ((i + chiArrPos) >= numFreqs)
                    break;
                outputVal = chiArr[chiArrPos + i] * cArray[tempArrPos + i];
                outputArray[outputArrPos + i] = MAX(outputArray[outputArrPos + i], outputVal);
            }

            chiArrPos += stepSize;
            outputArrPos += stepSize;
            __syncthreads();
        }
    }
}




//This function generates and sums random features for a Conv1d Maxpool-type kernel.
template <typename T>
const char *conv1dMaxpoolFeatureGen(const int8_t *radem, const T xdata[],
            const T chiArr[], double *outputArray, const int32_t *seqlengths,
            int xdim0, int xdim1, int xdim2, int numFreqs,
            int convWidth, int paddedBufferSize, int rademShape2){

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
    int stepSize = MIN(MAX_BASE_LEVEL_TRANSFORM, paddedBufferSize);
    int log2N = log2(paddedBufferSize);

    int numRepeats = (numFreqs + paddedBufferSize - 1) / paddedBufferSize;

    convMaxpoolFeatureGenKernel<T><<<xdim0, stepSize / 2, stepSize * sizeof(T)>>>(xdata, featureArray,
        outputArray, chiArr, radem, paddedBufferSize, log2N, numFreqs, xdim1, xdim2,
        numRepeats, normConstant, convWidth, seqlengths, rademShape2);
    
    cudaFree(featureArray);
    return "no_error";
}
//Explicitly instantiate so wrapper can use.
template const char *conv1dMaxpoolFeatureGen<float>(const int8_t *radem, const float *xdata,
            const float *chiArr, double *outputArray, const int32_t *seqlengths,
            int xdim0, int xdim1, int xdim2, int numFreqs,
            int convWidth, int paddedBufferSize, int rademShape2);
template const char *conv1dMaxpoolFeatureGen<double>(const int8_t *radem, const double *xdata,
            const double *chiArr, double *outputArray, const int32_t *seqlengths,
            int xdim0, int xdim1, int xdim2, int numFreqs,
            int convWidth, int paddedBufferSize, int rademShape2);
