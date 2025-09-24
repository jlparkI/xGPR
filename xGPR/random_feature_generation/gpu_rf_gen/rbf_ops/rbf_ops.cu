/* Copyright (C) 2025 Jonathan Parkinson
*/
// C++ headers
#include <stdio.h>
#include <stdint.h>
#include <math.h>

// Library headers
#include <cuda.h>
#include <cuda_runtime.h>

// Project headers
#include "../shared_constants.h"
#include "../sharedmem.h"
#include "rbf_ops.h"


namespace nb = nanobind;

template <typename T>
__global__ void rbfFeatureGenKernel(const T origData[], T cArray[],
        double *output_array, const T chi_arr[], const int8_t *radem,
        int padded_buffer_size, int log2N, int num_freqs, int input_elements_per_row,
        int nrepeats, int rademShape2, T norm_constant,
        double scaling_constant){
    int stepSize = MIN(padded_buffer_size, MAX_BASE_LEVEL_TRANSFORM);

    SharedMemory<T> shared;
    T *s_data = shared.getPointer();
    int spacing, pos = threadIdx.x;
    int lo, id1, id2;
    int temp_arr_pos, chi_arr_pos = 0;
    int input_arr_pos = (blockIdx.x * input_elements_per_row);
    int output_arr_pos = (blockIdx.x * num_freqs * 2);
    T y, output_val;
    const int8_t *rademPtr = radem;

    //Run over the number of repeats required to generate the random
    //features.
    for (int rep = 0; rep < nrepeats; rep++){
        temp_arr_pos = (blockIdx.x << log2N);

        //Copy original data into the temporary array.
        for (int i = threadIdx.x; i < padded_buffer_size; i += blockDim.x){
            if (i < input_elements_per_row)
                cArray[i + temp_arr_pos] = origData[i + input_arr_pos];
            else
                cArray[i + temp_arr_pos] = 0;
        }

        //Run over three repeats for the SORF procedure.
        for (int sorfRep = 0; sorfRep < 3; sorfRep++){
            rademPtr = radem + padded_buffer_size * rep + sorfRep * rademShape2;
            temp_arr_pos = (blockIdx.x << log2N);

            for (int hStep = 0; hStep < padded_buffer_size; hStep+=stepSize){
                for (int i = threadIdx.x; i < stepSize; i += blockDim.x)
                    s_data[i] = cArray[i + temp_arr_pos];

                __syncthreads();

                //Multiply by the diagonal array here.
                for (int i = threadIdx.x; i < stepSize; i += blockDim.x)
                    s_data[i] = s_data[i] * rademPtr[i] * norm_constant;

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
                    cArray[i + temp_arr_pos] = s_data[i];

                temp_arr_pos += stepSize;
                __syncthreads();
            }

            //A less efficient global memory procedure to complete the FHT
            //for long arrays.
            if (padded_buffer_size > MAX_BASE_LEVEL_TRANSFORM){
                temp_arr_pos = (blockIdx.x << log2N);

                for (int spacing = stepSize; spacing < padded_buffer_size; spacing <<= 1){

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
        //activation function, and populate the output array. Note that
        //we multiply by 2 in the output array position since two
        //features are generated for each frequency sampled.
        temp_arr_pos = (blockIdx.x << log2N);

        for (int i = threadIdx.x; i < padded_buffer_size; i += blockDim.x){
            if ((i + chi_arr_pos) >= num_freqs)
                break;
            output_val = chi_arr[chi_arr_pos + i] * cArray[temp_arr_pos + i];
            output_array[output_arr_pos + 2 * i] = scaling_constant * cos(output_val);
            output_array[output_arr_pos + 2 * i + 1] = scaling_constant * sin(output_val);
        }

        chi_arr_pos += padded_buffer_size;
        output_arr_pos += 2 * padded_buffer_size;
        __syncthreads();

    }
}




template <typename T>
__global__ void rbfFeatureGradKernel(const T origData[], T cArray[],
double *output_array, const T chi_arr[], const int8_t *radem,
int padded_buffer_size, int log2N, int num_freqs, int input_elements_per_row,
int nrepeats, int rademShape2, T norm_constant,
double scaling_constant, double *gradient, T sigma){
    int stepSize = MIN(padded_buffer_size, MAX_BASE_LEVEL_TRANSFORM);

    SharedMemory<T> shared;
    T *s_data = shared.getPointer();
    int spacing, pos = threadIdx.x;
    int lo, id1, id2;
    int temp_arr_pos, chi_arr_pos = 0;
    int input_arr_pos = (blockIdx.x * input_elements_per_row);
    int output_arr_pos = (blockIdx.x * num_freqs * 2);
    T y, output_val;
    const int8_t *rademPtr = radem;

    //Run over the number of repeats required to generate the random
    //features.
    for (int rep = 0; rep < nrepeats; rep++){
        temp_arr_pos = (blockIdx.x << log2N);

        //Copy original data into the temporary array.
        for (int i = threadIdx.x; i < padded_buffer_size; i += blockDim.x){
            if (i < input_elements_per_row)
                cArray[i + temp_arr_pos] = origData[i + input_arr_pos];
            else
                cArray[i + temp_arr_pos] = 0;
        }

        //Run over three repeats for the SORF procedure.
        for (int sorfRep = 0; sorfRep < 3; sorfRep++){
            rademPtr = radem + padded_buffer_size * rep + sorfRep * rademShape2;
            temp_arr_pos = (blockIdx.x << log2N);

            for (int hStep = 0; hStep < padded_buffer_size; hStep+=stepSize){
                for (int i = threadIdx.x; i < stepSize; i += blockDim.x)
                    s_data[i] = cArray[i + temp_arr_pos];

                __syncthreads();

                //Multiply by the diagonal array here.
                for (int i = threadIdx.x; i < stepSize; i += blockDim.x)
                    s_data[i] = s_data[i] * rademPtr[i] * norm_constant;

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
                    cArray[i + temp_arr_pos] = s_data[i];

                temp_arr_pos += stepSize;
                __syncthreads();
            }

            //A less efficient global memory procedure to complete the FHT
            //for long arrays.
            if (padded_buffer_size > MAX_BASE_LEVEL_TRANSFORM){
                temp_arr_pos = (blockIdx.x << log2N);

                for (int spacing = stepSize; spacing < padded_buffer_size; spacing <<= 1){

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
        //activation function, and populate the output array. Note that
        //we multiply by 2 in the output array position since two
        //features are generated for each frequency sampled.
        temp_arr_pos = (blockIdx.x << log2N);

        for (int i = threadIdx.x; i < padded_buffer_size; i += blockDim.x){
            if ((i + chi_arr_pos) >= num_freqs)
                break;
            output_val = chi_arr[chi_arr_pos + i] * cArray[temp_arr_pos + i];
            double prod_val = output_val * sigma;
            output_array[output_arr_pos + 2 * i] = scaling_constant * cos(prod_val);
            output_array[output_arr_pos + 2 * i + 1] = scaling_constant * sin(prod_val);
            gradient[output_arr_pos + 2 * i] = -scaling_constant * sin(prod_val) * output_val;
            gradient[output_arr_pos + 2 * i + 1] = scaling_constant * cos(prod_val) * output_val;
        }

        chi_arr_pos += padded_buffer_size;
        output_arr_pos += 2 * padded_buffer_size;
        __syncthreads();

    }
}




template <typename T>
int RBFFeatureGen(
nb::ndarray<const T, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> input_arr,
nb::ndarray<double, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> output_arr,
nb::ndarray<const int8_t, nb::shape<3,1,-1>, nb::device::cuda, nb::c_contig> radem,
nb::ndarray<const T, nb::shape<-1>, nb::device::cuda, nb::c_contig> chi_arr,
bool fit_intercept) {

    // Perform safety checks. Any exceptions thrown here are handed off to Python
    // by the Nanobind wrapper. We do not expect the user to see these because
    // the Python code will always ensure inputs are correct -- these are a failsafe
    // -- so we do not need to provide detailed exception messages here.
    int zDim0 = input_arr.shape(0);
    int zDim1 = input_arr.shape(1);
    size_t num_rffs = output_arr.shape(1);
    size_t num_freqs = chi_arr.shape(0);
    double num_freqsFlt = num_freqs;

    const T *input_ptr = input_arr.data();
    double *output_ptr = output_arr.data();
    const T *chi_ptr = chi_arr.data();
    const int8_t *rademPtr = radem.data();

    if (input_arr.shape(0) == 0 || output_arr.shape(0) != input_arr.shape(0))
        throw std::runtime_error("no datapoints");
    if (num_rffs < 2 || (num_rffs & 1) != 0)
        throw std::runtime_error("last dim of output must be even number");
    if ( (2 * num_freqs) != num_rffs || num_freqs > radem.shape(2) )
        throw std::runtime_error("incorrect number of rffs and or freqs.");

    double expectedNFreq = (zDim1 > 2) ? static_cast<double>(zDim1) : 2.0;
    double log2Freqs = std::log2(expectedNFreq);
    log2Freqs = std::ceil(log2Freqs);
    int padded_buffer_size = std::pow(2, log2Freqs);

    if (radem.shape(2) % padded_buffer_size != 0)
        throw std::runtime_error("incorrect number of rffs and or freqs.");

    T rbfNormConstant;

    if (fit_intercept)
        rbfNormConstant = std::sqrt(1.0 / (num_freqsFlt - 0.5));
    else
        rbfNormConstant = std::sqrt(1.0 / num_freqsFlt);

    //This is the Hadamard normalization constant.
    T norm_constant = log2(padded_buffer_size) / 2;
    norm_constant = 1 / pow(2, norm_constant);
    int num_repeats = (num_freqs + padded_buffer_size - 1) / padded_buffer_size;
    int stepSize = MIN(MAX_BASE_LEVEL_TRANSFORM, padded_buffer_size);
    int log2N = log2(padded_buffer_size);

    T *feature_array;
    if (cudaMalloc(&feature_array, sizeof(T) * zDim0 * padded_buffer_size) != cudaSuccess) {
        cudaFree(feature_array);
        throw std::runtime_error("out of memory on cuda");
        return 1;
    };

    rbfFeatureGenKernel<T><<<zDim0, stepSize / 2, stepSize * sizeof(T)>>>(input_ptr,
            feature_array, output_ptr, chi_ptr, rademPtr, padded_buffer_size, log2N, num_freqs, zDim1,
            num_repeats, radem.shape(2), norm_constant, rbfNormConstant);

    cudaFree(feature_array);
    return 0;
}
template int RBFFeatureGen<double>(
nb::ndarray<const double, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> input_arr,
nb::ndarray<double, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> output_arr,
nb::ndarray<const int8_t, nb::shape<3,1,-1>, nb::device::cuda, nb::c_contig> radem,
nb::ndarray<const double, nb::shape<-1>, nb::device::cuda, nb::c_contig> chi_arr,
bool fit_intercept);
template int RBFFeatureGen<float>(
nb::ndarray<const float, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> input_arr,
nb::ndarray<double, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> output_arr,
nb::ndarray<const int8_t, nb::shape<3,1,-1>, nb::device::cuda, nb::c_contig> radem,
nb::ndarray<const float, nb::shape<-1>, nb::device::cuda, nb::c_contig> chi_arr,
bool fit_intercept);


//it in a separate array.
template <typename T>
int RBFFeatureGrad(
nb::ndarray<const T, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> input_arr,
nb::ndarray<double, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> output_arr,
nb::ndarray<double, nb::shape<-1,-1,1>, nb::device::cuda, nb::c_contig> grad_arr,
nb::ndarray<const int8_t, nb::shape<3,1,-1>, nb::device::cuda, nb::c_contig> radem,
nb::ndarray<const T, nb::shape<-1>, nb::device::cuda, nb::c_contig> chi_arr,
float sigma, bool fit_intercept) {

    // Perform safety checks. Any exceptions thrown here are handed off to Python
    // by the Nanobind wrapper. We do not expect the user to see these because
    // the Python code will always ensure inputs are correct -- these are a failsafe
    // -- so we do not need to provide detailed exception messages here.
    int zDim0 = input_arr.shape(0);
    int zDim1 = input_arr.shape(1);
    size_t num_rffs = output_arr.shape(1);
    size_t num_freqs = chi_arr.shape(0);
    double num_freqsFlt = num_freqs;

    const T *input_ptr = input_arr.data();
    double *output_ptr = output_arr.data();
    double *gradient_ptr = grad_arr.data();
    const T *chi_ptr = chi_arr.data();
    const int8_t *rademPtr = radem.data();

    if (input_arr.shape(0) == 0 || output_arr.shape(0) != input_arr.shape(0))
        throw std::runtime_error("no datapoints");
    if (num_rffs < 2 || (num_rffs & 1) != 0)
        throw std::runtime_error("last dim of output must be even number");
    if ( (2 * num_freqs) != num_rffs || num_freqs > radem.shape(2) )
        throw std::runtime_error("incorrect number of rffs and or freqs.");
    if (grad_arr.shape(0) != output_arr.shape(0) || grad_arr.shape(1) != output_arr.shape(1))
        throw std::runtime_error("Wrong array sizes.");

    double expectedNFreq = (zDim1 > 2) ? static_cast<double>(zDim1) : 2.0;
    double log2Freqs = std::log2(expectedNFreq);
    log2Freqs = std::ceil(log2Freqs);
    int padded_buffer_size = std::pow(2, log2Freqs);

    if (radem.shape(2) % padded_buffer_size != 0)
        throw std::runtime_error("incorrect number of rffs and or freqs.");

    T rbfNormConstant;

    if (fit_intercept)
        rbfNormConstant = std::sqrt(1.0 / (num_freqsFlt - 0.5));
    else
        rbfNormConstant = std::sqrt(1.0 / num_freqsFlt);


    //This is the Hadamard normalization constant.
    T norm_constant = log2(padded_buffer_size) / 2;
    norm_constant = 1 / pow(2, norm_constant);
    int num_repeats = (num_freqs + padded_buffer_size - 1) / padded_buffer_size;
    int stepSize = MIN(MAX_BASE_LEVEL_TRANSFORM, padded_buffer_size);
    int log2N = log2(padded_buffer_size);

    T *feature_array;
    if (cudaMalloc(&feature_array, sizeof(T) * zDim0 * padded_buffer_size) != cudaSuccess) {
        cudaFree(feature_array);
        throw std::runtime_error("out of memory on cuda");
        return 1;
    };

    rbfFeatureGradKernel<T><<<zDim0, stepSize / 2, stepSize * sizeof(T)>>>(input_ptr,
            feature_array, output_ptr, chi_ptr, rademPtr, padded_buffer_size, log2N, num_freqs, zDim1,
            num_repeats, radem.shape(2), norm_constant, rbfNormConstant, gradient_ptr,
            sigma);

    cudaFree(feature_array);
    return 0;
}
template int RBFFeatureGrad<double>(
nb::ndarray<const double, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> input_arr,
nb::ndarray<double, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> output_arr,
nb::ndarray<double, nb::shape<-1,-1,1>, nb::device::cuda, nb::c_contig> grad_arr,
nb::ndarray<const int8_t, nb::shape<3,1,-1>, nb::device::cuda, nb::c_contig> radem,
nb::ndarray<const double, nb::shape<-1>, nb::device::cuda, nb::c_contig> chi_arr,
float sigma, bool fit_intercept);
template int RBFFeatureGrad<float>(
nb::ndarray<const float, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> input_arr,
nb::ndarray<double, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> output_arr,
nb::ndarray<double, nb::shape<-1,-1,1>, nb::device::cuda, nb::c_contig> grad_arr,
nb::ndarray<const int8_t, nb::shape<3,1,-1>, nb::device::cuda, nb::c_contig> radem,
nb::ndarray<const float, nb::shape<-1>, nb::device::cuda, nb::c_contig> chi_arr,
float sigma, bool fit_intercept);
