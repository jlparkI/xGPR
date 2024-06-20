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



//Generates the Conv kernel RBF features with the simplex modification of Reid et al. 2023.
//This single kernel loops over 1) kmers
//then 2) the number of repeats then inside that loop 3) the three diagonal
//matrix multiplications and fast Hadamard transforms before
//applying 4) simplex projection and 5) diagonal matmul before activation function.
template <typename T>
__global__ void convRBFFeatureGenSimplexKernel(const T origData[], T cArray[],
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
    T y, outputVal, bufferSum, simplexProjPrefactor, modifiedScaling = scalingConstant;

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
        double *gradient){

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
                gradient[outputArrPos + 2 * i] -= modifiedScaling * sin(outputVal) * outputVal;
                gradient[outputArrPos + 2 * i + 1] += modifiedScaling * cos(outputVal) * outputVal;
            }

            chiArrPos += paddedBufferSize;
            outputArrPos += 2 * paddedBufferSize;
            __syncthreads();
        }
    }
}




//Generates the Conv kernel RBF features together with the gradient but with
//the simplex modification of Reid et al. 2023. This single
//kernel loops over 1) kmers then 2) the number of repeats then inside that
//loop 3) the three diagonal matrix multiplications and fast Hadamard transforms
//before applying 4) simplex modification and 5) diagonal matmul before
//activation function.
template <typename T>
__global__ void convRBFFeatureGradSimplexKernel(const T origData[], T cArray[],
        double *outputArray, const T chiArr[], const int8_t *radem,
        int paddedBufferSize, int log2N, int numFreqs, int xDim1, int xDim2,
        int nRepeats, int rademShape2, T normConstant,
        double scalingConstant, int scalingType,
        int convWidth, const int32_t *seqlengths,
        double *gradient){

    int stepSize = MIN(paddedBufferSize, MAX_BASE_LEVEL_TRANSFORM);
    int colCutoff = seqlengths[blockIdx.x] - convWidth + 1;

    SharedMemory<T> shared;
    T *s_data = shared.getPointer();
    int spacing, pos = threadIdx.x;
    int lo, id1, id2;
    int tempArrPos, chiArrPos = 0, inputCutoff = xDim2 * convWidth;
    int inputArrPos = (blockIdx.x * xDim1 * xDim2);
    int outputArrPos = (blockIdx.x * numFreqs * 2);
    T y, outputVal, bufferSum, simplexProjPrefactor, modifiedScaling = scalingConstant;

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
                outputArray[outputArrPos + 2 * i] += modifiedScaling * cos(outputVal);
                outputArray[outputArrPos + 2 * i + 1] += modifiedScaling * sin(outputVal);
                gradient[outputArrPos + 2 * i] -= modifiedScaling * sin(outputVal) * outputVal;
                gradient[outputArrPos + 2 * i + 1] += modifiedScaling * cos(outputVal) * outputVal;
            }

            chiArrPos += stepSize;
            outputArrPos += 2 * stepSize;
            __syncthreads();
        }
    }
}






//This function generates and sums random features for a Conv1d RBF-type kernel.
template <typename T>
const char *convRBFFeatureGen(const int8_t *radem, const T xdata[],
            const T chiArr[], double *outputArray, const int32_t *seqlengths,
            int xdim0, int xdim1, int xdim2, int numFreqs,
            int rademShape2, int convWidth, int paddedBufferSize,
            double scalingTerm, int scalingType,
            bool simplex){

    //This is the Hadamard normalization constant.
    T normConstant = log2(paddedBufferSize) / 2;
    normConstant = 1 / pow(2, normConstant);

    int numRepeats = (numFreqs + paddedBufferSize - 1) / paddedBufferSize;
    int stepSize = MIN(MAX_BASE_LEVEL_TRANSFORM, paddedBufferSize);
    int log2N = log2(paddedBufferSize);

    T *featureArray;
    if (cudaMalloc(&featureArray, sizeof(T) * xdim0 * paddedBufferSize) != cudaSuccess) {
            cudaFree(featureArray);
            return "Fatal malloc error";
    };

    if (!simplex){
        convRBFFeatureGenKernel<T><<<xdim0, stepSize / 2, stepSize * sizeof(T)>>>(xdata, featureArray,
            outputArray, chiArr, radem, paddedBufferSize, log2N, numFreqs, xdim1, xdim2,
            numRepeats, rademShape2, normConstant, scalingTerm, scalingType, convWidth, seqlengths);
    }
    else{
        convRBFFeatureGenSimplexKernel<T><<<xdim0, stepSize / 2, stepSize * sizeof(T)>>>(xdata, featureArray,
            outputArray, chiArr, radem, paddedBufferSize, log2N, numFreqs, xdim1, xdim2,
            numRepeats, rademShape2, normConstant, scalingTerm, scalingType, convWidth, seqlengths);
    }

    cudaFree(featureArray);
    return "no_error";
}
//Explicitly instantiate so wrapper can use.
template const char *convRBFFeatureGen<float>(const int8_t *radem,
            const float xdata[], const float chiArr[], double *outputArray,
            const int32_t *seqlengths, int xdim0, int xdim1, int xdim2, int numFreqs,
            int rademShape2, int convWidth, int paddedBufferSize,
            double scalingTerm, int scalingType, bool simplex);
template const char *convRBFFeatureGen<double>(const int8_t *radem,
            const double xdata[], const double chiArr[], double *outputArray,
            const int32_t *seqlengths, int xdim0, int xdim1, int xdim2, int numFreqs,
            int rademShape2, int convWidth, int paddedBufferSize,
            double scalingTerm, int scalingType, bool simplex);



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
            double scalingTerm, int scalingType,
            bool simplex){
    //This is the Hadamard normalization constant.
    T normConstant = log2(paddedBufferSize) / 2;
    normConstant = 1 / pow(2, normConstant);

    int numRepeats = (numFreqs + paddedBufferSize - 1) / paddedBufferSize;
    int stepSize = MIN(MAX_BASE_LEVEL_TRANSFORM, paddedBufferSize);
    int log2N = log2(paddedBufferSize);

    T *featureArray;
    if (cudaMalloc(&featureArray, sizeof(T) * xdim0 * paddedBufferSize) != cudaSuccess) {
            cudaFree(featureArray);
            return "Fatal malloc error";
    };

    if (!simplex){
        convRBFFeatureGradKernel<T><<<xdim0, stepSize / 2, stepSize * sizeof(T)>>>(xdata, featureArray,
            outputArray, chiArr, radem, paddedBufferSize, log2N, numFreqs, xdim1, xdim2,
            numRepeats, rademShape2, normConstant, scalingTerm, scalingType, convWidth, seqlengths,
            gradientArray);
    }
    else{
        convRBFFeatureGradSimplexKernel<T><<<xdim0, stepSize / 2, stepSize * sizeof(T)>>>(xdata, featureArray,
            outputArray, chiArr, radem, paddedBufferSize, log2N, numFreqs, xdim1, xdim2,
            numRepeats, rademShape2, normConstant, scalingTerm, scalingType, convWidth, seqlengths,
            gradientArray);
    }

    cudaFree(featureArray);
    return "no_error";
}
//Explicitly instantiate so wrapper can use.
template const char *convRBFFeatureGrad<float>(const int8_t *radem,
            const float xdata[], const float chiArr[], double *outputArray,
            const int32_t *seqlengths, double *gradientArray, double sigma,
            int xdim0, int xdim1, int xdim2, int numFreqs,
            int rademShape2, int convWidth, int paddedBufferSize,
            double scalingTerm, int scalingType, bool simplex);
template const char *convRBFFeatureGrad<double>(const int8_t *radem,
            const double xdata[], const double chiArr[], double *outputArray,
            const int32_t *seqlengths, double *gradientArray, double sigma,
            int xdim0, int xdim1, int xdim2, int numFreqs,
            int rademShape2, int convWidth, int paddedBufferSize,
            double scalingTerm, int scalingType, bool simplex);
