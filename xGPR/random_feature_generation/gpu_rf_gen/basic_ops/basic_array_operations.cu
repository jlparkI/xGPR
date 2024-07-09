/*
* Contains tools for basic array operations on GPU -- fast hadamard transforms
* and SRHT. Note that many operations here assume specific dimensions
* of the input array are a power of 2. The Cython wrapper checks this, so do
* not call these routines OUTSIDE of the Cython wrapper -- use the Cython wrapper.
*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <math.h>
#include "../shared_constants.h"
#include "basic_array_operations.h"
#include "../sharedmem.h"


namespace nb = nanobind;


//Uses shared memory to perform a reasonably efficient single kernel
//transform up to MAX_BASE_LEVEL_TRANSFORM, then uses a somewhat
//less efficient global memory procedure to complete the
//transform if needed. Performs the transform on a single
//vector of the input.
template <typename T>
__global__ void hadamardTransform(T cArray[], int N, int log2N){
    int stepSize = MIN(N, MAX_BASE_LEVEL_TRANSFORM);
    int nRepeats = N / stepSize;

    SharedMemory<T> shared;
    T *s_data = shared.getPointer();
    int spacing, pos = threadIdx.x;
    int lo, id1, id2;
    T *src_ptr = cArray + (blockIdx.x << log2N);
    T y;

    for (int rep = 0; rep < nRepeats; rep++){
        for (int i = threadIdx.x; i < stepSize; i += blockDim.x)
            s_data[i] = src_ptr[i];

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
            src_ptr[i] = s_data[i];
        __syncthreads();
        src_ptr += stepSize;
    }

    if (N > MAX_BASE_LEVEL_TRANSFORM){
        src_ptr = cArray + (blockIdx.x << log2N);

        for (int spacing = stepSize; spacing < N; spacing <<= 1){
            __syncthreads();

            for (int k = 0; k < N; k += (spacing << 1)){
                for (int i = threadIdx.x; i < spacing; i += blockDim.x){
                    id1 = i+k;
                    id2 = id1 + spacing;
                    y = src_ptr[id2];
                    src_ptr[id2] = src_ptr[id1] - y;
                    src_ptr[id1] += y;
                }
            }
        }
    }
}


//Performs a Hadamard transform on a 2d array after first multiplying
//by a diagonal array with entries populated from a Rademacher distribution.
//This is intended for use with SRHT and related procedures.
template <typename T>
__global__ void hadamardTransformRadMult(T cArray[], int N, int log2N,
        const int8_t *radem, T normConstant){
    int stepSize = MIN(N, MAX_BASE_LEVEL_TRANSFORM);
    int nRepeats = N / stepSize;

    SharedMemory<T> shared;
    T *s_data = shared.getPointer();
    int spacing, pos = threadIdx.x;
    int lo, id1, id2;
    T *src_ptr = cArray + (blockIdx.x << log2N);
    T y;

    const int8_t *rademPtr = radem;

    for (int rep = 0; rep < nRepeats; rep++){
        for (int i = threadIdx.x; i < stepSize; i += blockDim.x)
            s_data[i] = src_ptr[i];
    
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
            src_ptr[i] = s_data[i];
        __syncthreads();
        src_ptr += stepSize;
    }

    if (N > MAX_BASE_LEVEL_TRANSFORM){
        src_ptr = cArray + (blockIdx.x << log2N);

        for (int spacing = stepSize; spacing < N; spacing <<= 1){
            __syncthreads();

            for (int k = 0; k < N; k += (spacing << 1)){
                for (int i = threadIdx.x; i < spacing; i += blockDim.x){
                    id1 = i+k;
                    id2 = id1 + spacing;
                    y = src_ptr[id2];
                    src_ptr[id2] = src_ptr[id1] - y;
                    src_ptr[id1] += y;
                }
            }
        }
    }
}


//Performs an elementwise multiplication of a [c,M,P] array against the
//[N,M,P] input array or a [P] array against the [N,P] input array.
//Note that the last dimensions of these must be the
//same, and this function does not check this -- caller must check. Note that
//we multiply by the Hadamard normalization constant here.
template <typename T>
__global__ void diagonalRademMultiply(T cArray[], const int8_t *rademArray,
			int numElementsPerRow, int numElements, T normConstant)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int position = tid % numElementsPerRow;

    if (tid < numElements)
        cArray[tid] = cArray[tid] * rademArray[position] * normConstant;
}


//Performs an elementwise multiplication of a row of a [3,1,P x S] array against the
//float [N,M,S] input array. Note that the dimensions must be checked before calling
//-- done by the wrapper -- and that only S elements of the appropriate row of
//the [3, 1, P x S] array are used.
template <typename T>
__global__ void conv1dDiagonalRademMultiply(T cArray[],
            const int8_t *rademArray,
			int dim2, int startPosition, int numElements,
            T normConstant)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int position = startPosition + (tid & (dim2 - 1));
    
    if (tid < numElements)
        cArray[tid] = cArray[tid] * rademArray[position] * normConstant;
}




//Performs an unnormalized fast Hadamard transform over the last
//dimension of the input array.
template <typename T>
int cudaHTransform(nb::ndarray<T, nb::shape<-1,-1>, nb::device::cuda,
        nb::c_contig> inputArr){

    // Perform safety checks. Any exceptions thrown here are handed off to Python
    // by the Nanobind wrapper. We do not expect the user to see these because
    // the Python code will always ensure inputs are correct -- these are a failsafe
    // -- so we do not need to provide detailed exception messages here.
    int zDim0 = inputArr.shape(0);
    int zDim1 = inputArr.shape(1);
    T *inputPtr = inputArr.data();
    
    if (inputArr.shape(0) == 0)
        throw std::runtime_error("no datapoints");
    if (inputArr.shape(2) < 2)
        throw std::runtime_error("last dim not power of 2 > 1");
    if ((zDim1 & (zDim1 - 1)) != 0)
        throw std::runtime_error("last dim not power of 2");


    int stepSize, log2N;
    stepSize = MIN(MAX_BASE_LEVEL_TRANSFORM, zDim1);
    log2N = log2(zDim1);

    hadamardTransform<T><<<zDim0, stepSize / 2,
                    stepSize * sizeof(T)>>>(inputPtr, zDim1, log2N);

    // Update this to add error code handling.
    return 0;
}
//Instantiate templates explicitly so wrapper can use.
template int cudaHTransform<double>(nb::ndarray<double, nb::shape<-1, -1>, nb::device::cuda,
        nb::c_contig> inputArr);
template int cudaHTransform<float>(nb::ndarray<float, nb::shape<-1, -1>, nb::device::cuda,
        nb::c_contig> inputArr);



//Performs the first two steps of SRHT (HD)
template <typename T>
int cudaSRHT2d(nb::ndarray<T, nb::shape<-1,-1>, nb::device::cuda,
        nb::c_contig> inputArr,
        nb::ndarray<const int8_t, nb::shape<-1>, nb::device::cuda,
        nb::c_contig> radem,
        int numThreads){
    // Perform safety checks. Any exceptions thrown here are handed off to Python
    // by the Nanobind wrapper. We do not expect the user to see these because
    // the Python code will always ensure inputs are correct -- these are a failsafe
    // -- so we do not need to provide detailed exception messages here.
    int zDim0 = inputArr.shape(0);
    int zDim1 = inputArr.shape(1);
    T *inputPtr = inputArr.data();
    const int8_t *rademPtr = radem.data();
    
    if (inputArr.shape(0) == 0)
        throw std::runtime_error("no datapoints");
    if (inputArr.shape(1) != radem.shape(0))
        throw std::runtime_error("wrong array sizes");
    if (inputArr.shape(2) < 2)
        throw std::runtime_error("last dim not power of 2 > 1");
    if ((zDim1 & (zDim1 - 1)) != 0)
        throw std::runtime_error("last dim not power of 2");

    //This is the Hadamard normalization constant.
    T normConstant = log2(zDim1) / 2;
    normConstant = 1 / pow(2, normConstant);
    int stepSize, log2N;
    stepSize = MIN(MAX_BASE_LEVEL_TRANSFORM, zDim1);
    log2N = log2(zDim1);


    //cudaProfilerStart();
    hadamardTransformRadMult<T><<<zDim0, stepSize / 2,
        stepSize * sizeof(T)>>>(inputPtr, zDim1, log2N,
                    rademPtr, normConstant);


    //cudaProfilerStop();
    return 0;
}
//Instantiate templates explicitly so wrapper can use.
template int cudaSRHT2d<double>(nb::ndarray<double, nb::shape<-1,-1>, nb::device::cuda,
        nb::c_contig> inputArr,
        nb::ndarray<const int8_t, nb::shape<-1>, nb::device::cuda,
        nb::c_contig> radem,
        int numThreads);
template int cudaSRHT2d<float>(nb::ndarray<float, nb::shape<-1,-1>, nb::device::cuda,
        nb::c_contig> inputArr,
        nb::ndarray<const int8_t, nb::shape<-1>, nb::device::cuda,
        nb::c_contig> radem,
        int numThreads);
