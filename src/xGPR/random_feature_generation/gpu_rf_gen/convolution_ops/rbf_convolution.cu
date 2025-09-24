/* Copyright (C) 2025 Jonathan Parkinson
*/
// C++ headers
#include <stdint.h>
#include <math.h>

// Library headers
#include <cuda.h>
#include <cuda_runtime.h>

// Project headers
#include "../shared_constants.h"
#include "../sharedmem.h"
#include "rbf_convolution.h"



template <typename T>
__global__ void convRBFFeatureGenKernel(const T origData[], T cArray[],
double *output_array, const T chi_arr[], const int8_t *radem,
int padded_buffer_size, int log2N, int num_freqs, int xDim1, int xDim2,
int nrepeats, int rademShape2, T norm_constant,
double scaling_constant, int scaling_type,
int conv_width, const int32_t *seqlengths){

    int step_size = MIN(padded_buffer_size, MAX_BASE_LEVEL_TRANSFORM);
    int colCutoff = seqlengths[blockIdx.x] - conv_width + 1;

    SharedMemory<T> shared;
    T *s_data = shared.getPointer();
    int spacing, pos = threadIdx.x;
    int lo, id1, id2;
    int tempArrPos, chi_arr_pos = 0, inputCutoff = xDim2 * conv_width;
    int input_arrPos = (blockIdx.x * xDim1 * xDim2);
    int output_arr_pos = (blockIdx.x * num_freqs * 2);
    T y, output_val, modified_scaling = scaling_constant;

    const int8_t *radem_ptr = radem;

    switch (scaling_type){
        case 0:
            break;
        case 1:
            modified_scaling = modified_scaling / sqrt( (double) colCutoff);
            break;
        case 2:
            modified_scaling = modified_scaling / (double) colCutoff;
            break;
    }

    //Loop over the kmers in this stretch.
    for (int kmer = 0; kmer < colCutoff; kmer++){
        chi_arr_pos = 0;
        output_arr_pos = (blockIdx.x * num_freqs * 2);
        input_arrPos = (blockIdx.x * xDim1 * xDim2) + kmer * xDim2;

        //Run over the number of repeats required to generate the random
        //features.
        for (int rep = 0; rep < nrepeats; rep++){
            tempArrPos = (blockIdx.x << log2N);

            //Copy original data into the temporary array.
            for (int i = threadIdx.x; i < padded_buffer_size; i += blockDim.x){
                if (i < inputCutoff)
                    cArray[i + tempArrPos] = origData[i + input_arrPos];
                else
                    cArray[i + tempArrPos] = 0;
            }

            //Run over three repeats for the SORF procedure.
            for (int sorfRep = 0; sorfRep < 3; sorfRep++){
                radem_ptr = radem + padded_buffer_size * rep + sorfRep * rademShape2;
                tempArrPos = (blockIdx.x << log2N);

                for (int hStep = 0; hStep < padded_buffer_size; hStep+=step_size){
                    for (int i = threadIdx.x; i < step_size; i += blockDim.x)
                        s_data[i] = cArray[i + tempArrPos];

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
                        cArray[i + tempArrPos] = s_data[i];

                    tempArrPos += step_size;
                    __syncthreads();
                }

                //A less efficient global memory procedure to complete the FHT
                //for long arrays.
                if (padded_buffer_size > MAX_BASE_LEVEL_TRANSFORM){
                    tempArrPos = (blockIdx.x << log2N);

                    for (int spacing = step_size; spacing < padded_buffer_size; spacing <<= 1){

                        for (int k = 0; k < padded_buffer_size; k += (spacing << 1)){
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

            for (int i = threadIdx.x; i < padded_buffer_size; i += blockDim.x){
                if ((i + chi_arr_pos) >= num_freqs)
                    break;
                output_val = chi_arr[chi_arr_pos + i] * cArray[tempArrPos + i];
                output_array[output_arr_pos + 2 * i] += modified_scaling * cos(output_val);
                output_array[output_arr_pos + 2 * i + 1] += modified_scaling * sin(output_val);
            }

            chi_arr_pos += padded_buffer_size;
            output_arr_pos += 2 * padded_buffer_size;
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
double *output_array, const T chi_arr[], const int8_t *radem,
int padded_buffer_size, int log2N, int num_freqs, int xDim1, int xDim2,
int nrepeats, int rademShape2, T norm_constant,
double scaling_constant, int scaling_type,
int conv_width, const int32_t *seqlengths,
double *gradient, T sigma){

    int step_size = MIN(padded_buffer_size, MAX_BASE_LEVEL_TRANSFORM);
    int colCutoff = seqlengths[blockIdx.x] - conv_width + 1;

    SharedMemory<T> shared;
    T *s_data = shared.getPointer();
    int spacing, pos = threadIdx.x;
    int lo, id1, id2;
    int tempArrPos, chi_arr_pos = 0, inputCutoff = xDim2 * conv_width;
    int input_arrPos = (blockIdx.x * xDim1 * xDim2);
    int output_arr_pos = (blockIdx.x * num_freqs * 2);
    T y, output_val, modified_scaling = scaling_constant;

    const int8_t *radem_ptr = radem;

    switch (scaling_type){
        case 0:
            break;
        case 1:
            modified_scaling = modified_scaling / sqrt( (double) colCutoff);
            break;
        case 2:
            modified_scaling = modified_scaling / (double) colCutoff;
            break;
    }

    //Loop over the kmers in this stretch.
    for (int kmer = 0; kmer < colCutoff; kmer++){
        chi_arr_pos = 0;
        output_arr_pos = (blockIdx.x * num_freqs * 2);
        input_arrPos = (blockIdx.x * xDim1 * xDim2) + kmer * xDim2;

        //Run over the number of repeats required to generate the random
        //features.
        for (int rep = 0; rep < nrepeats; rep++){
            tempArrPos = (blockIdx.x << log2N);

            //Copy original data into the temporary array.
            for (int i = threadIdx.x; i < padded_buffer_size; i += blockDim.x){
                if (i < inputCutoff)
                    cArray[i + tempArrPos] = origData[i + input_arrPos];
                else
                    cArray[i + tempArrPos] = 0;
            }

            //Run over three repeats for the SORF procedure.
            for (int sorfRep = 0; sorfRep < 3; sorfRep++){
                radem_ptr = radem + padded_buffer_size * rep + sorfRep * rademShape2;
                tempArrPos = (blockIdx.x << log2N);

                for (int hStep = 0; hStep < padded_buffer_size; hStep+=step_size){
                    for (int i = threadIdx.x; i < step_size; i += blockDim.x)
                        s_data[i] = cArray[i + tempArrPos];

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
                        cArray[i + tempArrPos] = s_data[i];

                    tempArrPos += step_size;
                    __syncthreads();
                }

                //A less efficient global memory procedure to complete the FHT
                //for long arrays.
                if (padded_buffer_size > MAX_BASE_LEVEL_TRANSFORM){
                    tempArrPos = (blockIdx.x << log2N);

                    for (int spacing = step_size; spacing < padded_buffer_size; spacing <<= 1){

                        for (int k = 0; k < padded_buffer_size; k += (spacing << 1)){
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

            for (int i = threadIdx.x; i < padded_buffer_size; i += blockDim.x){
                if ((i + chi_arr_pos) >= num_freqs)
                    break;
                output_val = chi_arr[chi_arr_pos + i] * cArray[tempArrPos + i];
                double prod_val = output_val * sigma;
                output_array[output_arr_pos + 2 * i] += modified_scaling * cos(prod_val);
                output_array[output_arr_pos + 2 * i + 1] += modified_scaling * sin(prod_val);
                gradient[output_arr_pos + 2 * i] -= modified_scaling * sin(prod_val) * output_val;
                gradient[output_arr_pos + 2 * i + 1] += modified_scaling * cos(prod_val) * output_val;
            }

            chi_arr_pos += padded_buffer_size;
            output_arr_pos += 2 * padded_buffer_size;
            __syncthreads();
        }
    }
}






//This function generates and sums random features for a Conv1d RBF-type kernel.
template <typename T>
int convRBFFeatureGen(nb::ndarray<T, nb::shape<-1,-1,-1>, nb::device::cuda, nb::c_contig> input_arr,
nb::ndarray<double, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> output_arr,
nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cuda, nb::c_contig> radem,
nb::ndarray<T, nb::shape<-1>, nb::device::cuda, nb::c_contig> chi_arr,
nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> seqlengths,
int conv_width, int scaling_type) {

    // Perform safety checks. Any exceptions thrown here are handed off to Python
    // by the Nanobind wrapper. We do not expect the user to see these because
    // the Python code will always ensure inputs are correct -- these are a failsafe
    // -- so we do not need to provide detailed exception messages here.
    int zDim0 = input_arr.shape(0);
    int zDim1 = input_arr.shape(1);
    int zDim2 = input_arr.shape(2);
    size_t num_rffs = output_arr.shape(1);
    size_t num_freqs = chi_arr.shape(0);
    double scaling_term = std::sqrt(1.0 / static_cast<double>(num_freqs));

    T *inputPtr = input_arr.data();
    double *output_ptr = output_arr.data();
    T *chi_ptr = chi_arr.data();
    int8_t *radem_ptr = radem.data();
    int32_t *seqlengthsPtr = seqlengths.data();

    if (input_arr.shape(0) == 0 || output_arr.shape(0) != input_arr.shape(0))
        throw std::runtime_error("no datapoints");
    if (num_rffs < 2 || (num_rffs & 1) != 0)
        throw std::runtime_error("last dim of output must be even number");
    if ( (2 * num_freqs) != num_rffs || num_freqs > radem.shape(2) )
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


    //This is the Hadamard normalization constant.
    T norm_constant = log2(padded_buffer_size) / 2;
    norm_constant = 1 / pow(2, norm_constant);

    int numRepeats = (num_freqs + padded_buffer_size - 1) / padded_buffer_size;
    int step_size = MIN(MAX_BASE_LEVEL_TRANSFORM, padded_buffer_size);
    int log2N = log2(padded_buffer_size);

    T *feature_array;
    if (cudaMalloc(&feature_array, sizeof(T) * zDim0 * padded_buffer_size) != cudaSuccess) {
        cudaFree(slenCudaPtr);
        cudaFree(feature_array);
        throw std::runtime_error("Cuda is out of memory");
        return 1;
    };

    convRBFFeatureGenKernel<T><<<zDim0, step_size / 2, step_size * sizeof(T)>>>(inputPtr,
            feature_array, output_ptr, chi_ptr, radem_ptr, padded_buffer_size, log2N, num_freqs, zDim1, zDim2,
            numRepeats, radem.shape(2), norm_constant, scaling_term, scaling_type, conv_width,
            slenCudaPtr);

    cudaFree(slenCudaPtr);
    cudaFree(feature_array);
    return 0;
}
template int convRBFFeatureGen<double>(
nb::ndarray<double, nb::shape<-1,-1,-1>, nb::device::cuda, nb::c_contig> input_arr,
nb::ndarray<double, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> output_arr,
nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cuda, nb::c_contig> radem,
nb::ndarray<double, nb::shape<-1>, nb::device::cuda, nb::c_contig> chi_arr,
nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> seqlengths,
int conv_width, int scaling_type);
template int convRBFFeatureGen<float>(
nb::ndarray<float, nb::shape<-1,-1,-1>, nb::device::cuda, nb::c_contig> input_arr,
nb::ndarray<double, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> output_arr,
nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cuda, nb::c_contig> radem,
nb::ndarray<float, nb::shape<-1>, nb::device::cuda, nb::c_contig> chi_arr,
nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> seqlengths,
int conv_width, int scaling_type);



//This function generates and sums random features for an
//input array reshapedX of input type float WHILE also
//generating gradient information and storing this in
//a separate array. This gradient is only applicable
//in cases where all of the features share the same
//lengthscale; ARD-type kernels require a more complicated
//gradient calculation not implemented here.
template <typename T>
int convRBFFeatureGrad(nb::ndarray<T, nb::shape<-1,-1,-1>, nb::device::cuda, nb::c_contig> input_arr,
nb::ndarray<double, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> output_arr,
nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cuda, nb::c_contig> radem,
nb::ndarray<T, nb::shape<-1>, nb::device::cuda, nb::c_contig> chi_arr,
nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> seqlengths,
nb::ndarray<double, nb::shape<-1,-1,1>, nb::device::cuda, nb::c_contig> grad_arr,
double sigma, int conv_width, int scaling_type) {

    // Perform safety checks. Any exceptions thrown here are handed off to Python
    // by the Nanobind wrapper. We do not expect the user to see these because
    // the Python code will always ensure inputs are correct -- these are a failsafe
    // -- so we do not need to provide detailed exception messages here.
    int zDim0 = input_arr.shape(0);
    int zDim1 = input_arr.shape(1);
    int zDim2 = input_arr.shape(2);
    size_t num_rffs = output_arr.shape(1);
    size_t num_freqs = chi_arr.shape(0);
    double scaling_term = std::sqrt(1.0 / static_cast<double>(num_freqs));

    T *inputPtr = input_arr.data();
    double *output_ptr = output_arr.data();
    T *chi_ptr = chi_arr.data();
    int8_t *radem_ptr = radem.data();
    int32_t *seqlengthsPtr = seqlengths.data();
    double *gradient_ptr = grad_arr.data();

    if (input_arr.shape(0) == 0 || output_arr.shape(0) != input_arr.shape(0))
        throw std::runtime_error("no datapoints");
    if (num_rffs < 2 || (num_rffs & 1) != 0)
        throw std::runtime_error("last dim of output must be even number");
    if ( (2 * num_freqs) != num_rffs || num_freqs > radem.shape(2) )
        throw std::runtime_error("incorrect number of rffs and or freqs.");

    if (seqlengths.shape(0) != input_arr.shape(0))
        throw std::runtime_error("wrong array sizes");
    if (static_cast<int>(input_arr.shape(1)) < conv_width || conv_width <= 0)
        throw std::runtime_error("invalid conv_width");

    if (grad_arr.shape(0) != output_arr.shape(0) || grad_arr.shape(1) != output_arr.shape(1))
        throw std::runtime_error("wrong array sizes");

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




    //This is the Hadamard normalization constant.
    T norm_constant = log2(padded_buffer_size) / 2;
    norm_constant = 1 / pow(2, norm_constant);

    int numRepeats = (num_freqs + padded_buffer_size - 1) / padded_buffer_size;
    int step_size = MIN(MAX_BASE_LEVEL_TRANSFORM, padded_buffer_size);
    int log2N = log2(padded_buffer_size);

    T *feature_array;
    if (cudaMalloc(&feature_array, sizeof(T) * zDim0 * padded_buffer_size) != cudaSuccess) {
        cudaFree(slenCudaPtr);
        cudaFree(feature_array);
        throw std::runtime_error("Cuda is out of memory");
        return 1;
    };

    convRBFFeatureGradKernel<T><<<zDim0, step_size / 2, step_size * sizeof(T)>>>(inputPtr,
            feature_array, output_ptr, chi_ptr, radem_ptr, padded_buffer_size, log2N, num_freqs, zDim1, zDim2,
            numRepeats, radem.shape(2), norm_constant, scaling_term, scaling_type, conv_width,
            slenCudaPtr, gradient_ptr, sigma);

    cudaFree(slenCudaPtr);
    cudaFree(feature_array);
    return 0;
}
//Explicitly instantiate so wrapper can use.
template int convRBFFeatureGrad<double>(nb::ndarray<double, nb::shape<-1,-1,-1>, nb::device::cuda, nb::c_contig> input_arr,
nb::ndarray<double, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> output_arr,
nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cuda, nb::c_contig> radem,
nb::ndarray<double, nb::shape<-1>, nb::device::cuda, nb::c_contig> chi_arr,
nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> seqlengths,
nb::ndarray<double, nb::shape<-1,-1,1>, nb::device::cuda, nb::c_contig> grad_arr,
double sigma, int conv_width, int scaling_type);
template int convRBFFeatureGrad<float>(nb::ndarray<float, nb::shape<-1,-1,-1>, nb::device::cuda, nb::c_contig> input_arr,
nb::ndarray<double, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> output_arr,
nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cuda, nb::c_contig> radem,
nb::ndarray<float, nb::shape<-1>, nb::device::cuda, nb::c_contig> chi_arr,
nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> seqlengths,
nb::ndarray<double, nb::shape<-1,-1,1>, nb::device::cuda, nb::c_contig> grad_arr,
double sigma, int conv_width, int scaling_type);
