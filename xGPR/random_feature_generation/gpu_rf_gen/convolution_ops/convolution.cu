/* Copyright (C) 2025 Jonathan Parkinson
*/
// C++ headers
#include <math.h>
#include <stdint.h>

// Library headers
#include <cuda.h>
#include <cuda_runtime.h>

// Project headers
#include "../shared_constants.h"
#include "../sharedmem.h"
#include "convolution.h"

template <typename T>
__global__ void convMaxpoolFeatureGenKernel(const T origData[], T cArray[],
float *output_array, const T chi_arr[], const int8_t *radem,
int padded_buffer_size, int log2N, int num_freqs, int xDim1, int xDim2,
int nRepeats, T norm_constant, int conv_width,
const int32_t *seqlengths, int rademShape2){

    int step_size = MIN(padded_buffer_size, MAX_BASE_LEVEL_TRANSFORM);
    int col_cutoff = seqlengths[blockIdx.x] - conv_width + 1;

    SharedMemory<T> shared;
    T *s_data = shared.getPointer();
    int spacing, pos = threadIdx.x;
    int lo, id1, id2;
    int temp_arr_pos, chi_arr_pos = 0, inputCutoff = xDim2 * conv_width;
    int input_arr_pos = (blockIdx.x * xDim1 * xDim2);
    int output_arr_pos = (blockIdx.x * num_freqs);
    T y, outputVal;

    const int8_t *radem_ptr = radem;

    //Loop over the kmers in this stretch.
    for (int kmer = 0; kmer < col_cutoff; kmer++){
        chi_arr_pos = 0;
        output_arr_pos = (blockIdx.x * num_freqs);
        input_arr_pos = (blockIdx.x * xDim1 * xDim2) + kmer * xDim2;

        //Run over the number of repeats required to generate the random
        //features.
        for (int rep = 0; rep < nRepeats; rep++){
            temp_arr_pos = (blockIdx.x << log2N);

            //Copy original data into the temporary array.
            for (int i = threadIdx.x; i < padded_buffer_size; i += blockDim.x){
                if (i < inputCutoff)
                    cArray[i + temp_arr_pos] = origData[i + input_arr_pos];
                else
                    cArray[i + temp_arr_pos] = 0;
            }

            //Run over three repeats for the SORF procedure.
            for (int sorfRep = 0; sorfRep < 3; sorfRep++){
                radem_ptr = radem + padded_buffer_size * rep + sorfRep * rademShape2;
                temp_arr_pos = (blockIdx.x << log2N);

                for (int hStep = 0; hStep < padded_buffer_size; hStep+=step_size){
                    for (int i = threadIdx.x; i < step_size; i += blockDim.x)
                        s_data[i] = cArray[i + temp_arr_pos];

                    __syncthreads();

                    //Multiply by the diagonal array here.
                    for (int i = threadIdx.x; i < step_size; i += blockDim.x)
                        s_data[i] = s_data[i] * radem_ptr[i] * norm_constant;

                    radem_ptr += step_size;

                    id1 = (pos << 1);
                    id2 = id1 + 1;
                    __syncthreads();
                    y = s_data[id2];
                    s_data[id2] = s_data[id1] - y;
                    s_data[id1] += y;

                    for (spacing = 2; spacing < step_size; spacing <<= 1){
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

                    for (int i = threadIdx.x; i < step_size; i += blockDim.x)
                        cArray[i + temp_arr_pos] = s_data[i];

                    temp_arr_pos += step_size;
                    __syncthreads();
                }

                //A less efficient global memory procedure to complete the FHT
                //for long arrays.
                if (padded_buffer_size > MAX_BASE_LEVEL_TRANSFORM){
                    temp_arr_pos = (blockIdx.x << log2N);

                    for (int spacing = step_size; spacing < padded_buffer_size; spacing <<= 1){

                        for (int k = 0; k < padded_buffer_size; k += (spacing << 1)){
                            for (int i = threadIdx.x; i < spacing; i += blockDim.x){
                                id1 = i + k + temp_arr_pos;
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
            //activation function, and populate the output array.
            temp_arr_pos = (blockIdx.x << log2N);

            for (int i = threadIdx.x; i < padded_buffer_size; i += blockDim.x){
                if ((i + chi_arr_pos) >= num_freqs)
                    break;
                outputVal = chi_arr[chi_arr_pos + i] * cArray[temp_arr_pos + i];
                output_array[output_arr_pos + i] = MAX(output_array[output_arr_pos + i], outputVal);
            }

            chi_arr_pos += padded_buffer_size;
            output_arr_pos += padded_buffer_size;
            __syncthreads();
        }
    }
}



template <typename T>
int conv1dMaxpoolFeatureGen(nb::ndarray<T, nb::shape<-1,-1,-1>, nb::device::cuda, nb::c_contig> input_arr,
nb::ndarray<float, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> output_arr,
nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cuda, nb::c_contig> radem,
nb::ndarray<T, nb::shape<-1>, nb::device::cuda, nb::c_contig> chi_arr,
nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> seqlengths,
int conv_width) {

    // Perform safety checks. Any exceptions thrown here are handed off to Python
    // by the Nanobind wrapper. We do not expect the user to see these because
    // the Python code will always ensure inputs are correct -- these are a failsafe
    // -- so we do not need to provide detailed exception messages here.
    int zDim0 = input_arr.shape(0);
    int zDim1 = input_arr.shape(1);
    int zDim2 = input_arr.shape(2);
    size_t num_rffs = output_arr.shape(1);
    size_t num_freqs = chi_arr.shape(0);

    T *input_ptr = static_cast<T*>(input_arr.data());
    float *output_ptr = static_cast<float*>(output_arr.data());
    T *chi_ptr = static_cast<T*>(chi_arr.data());
    int8_t *radem_ptr = static_cast<int8_t*>(radem.data());
    int32_t *seqlengthsPtr = static_cast<int32_t*>(seqlengths.data());

    if (input_arr.shape(0) == 0 || output_arr.shape(0) != input_arr.shape(0))
        throw std::runtime_error("no datapoints");
    if (num_rffs < 2 || (num_rffs & 1) != 0)
        throw std::runtime_error("last dim of output must be even number");
    if (num_freqs != num_rffs || num_freqs > radem.shape(2) )
        throw std::runtime_error("incorrect number of rffs and or freqs.");

    if (seqlengths.shape(0) != input_arr.shape(0))
        throw std::runtime_error("wrong array sizes");
    if (static_cast<int>(input_arr.shape(1)) < conv_width || conv_width <= 0)
        throw std::runtime_error("invalid conv_width");

    double expectedNFreq = static_cast<double>(conv_width * input_arr.shape(2));
    expectedNFreq = MAX(expectedNFreq, 2);
    double log2Freqs = std::log2(expectedNFreq);
    log2Freqs = std::ceil(log2Freqs);
    int padded_buffer_size = std::pow(2, log2Freqs);

    if (radem.shape(2) % padded_buffer_size != 0)
        throw std::runtime_error("incorrect number of rffs and or freqs.");


    int32_t minSeqLength = 2147483647, maxSeqLength = 0;
    for (size_t i=0; i < seqlengths.shape(0); i++){
        if (seqlengths(i) > maxSeqLength)
            maxSeqLength = seqlengths(i);
        if (seqlengths(i) < minSeqLength)
            minSeqLength = seqlengths(i);
    }

    if (maxSeqLength > static_cast<int32_t>(input_arr.shape(1)) || minSeqLength < conv_width){
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




    int num_kmers = zDim1 - conv_width + 1;
    int num_elements = zDim0 * num_kmers * padded_buffer_size;

    T *feature_array;
    if (cudaMalloc(&feature_array, sizeof(T) * num_elements) != cudaSuccess) {
        cudaFree(slenCudaPtr);
        cudaFree(feature_array);
        throw std::runtime_error("Cuda is out of memory");
        return 1;
    };

    //This is the Hadamard normalization constant.
    T norm_constant = log2(padded_buffer_size) / 2;
    norm_constant = 1 / pow(2, norm_constant);
    int step_size = MIN(MAX_BASE_LEVEL_TRANSFORM, padded_buffer_size);
    int log2N = log2(padded_buffer_size);

    int num_repeats = (num_freqs + padded_buffer_size - 1) / padded_buffer_size;

    convMaxpoolFeatureGenKernel<T><<<zDim0, step_size / 2, step_size * sizeof(T)>>>(input_ptr,
            feature_array, output_ptr, chi_ptr, radem_ptr, padded_buffer_size, log2N, num_freqs,
            zDim1, zDim2, num_repeats, norm_constant, conv_width, slenCudaPtr, radem.shape(2));

    cudaFree(slenCudaPtr);
    cudaFree(feature_array);
    return 0;
}
//Explicitly instantiate so wrapper can use.
template int conv1dMaxpoolFeatureGen<double>(
nb::ndarray<double, nb::shape<-1,-1,-1>, nb::device::cuda, nb::c_contig> input_arr,
nb::ndarray<float, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> output_arr,
nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cuda, nb::c_contig> radem,
nb::ndarray<double, nb::shape<-1>, nb::device::cuda, nb::c_contig> chi_arr,
nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> seqlengths,
int conv_width);
template int conv1dMaxpoolFeatureGen<float>(
nb::ndarray<float, nb::shape<-1,-1,-1>, nb::device::cuda, nb::c_contig> input_arr,
nb::ndarray<float, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> output_arr,
nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cuda, nb::c_contig> radem,
nb::ndarray<float, nb::shape<-1>, nb::device::cuda, nb::c_contig> chi_arr,
nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> seqlengths,
int conv_width);
