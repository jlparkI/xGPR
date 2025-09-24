/* Copyright (C) 2025 Jonathan Parkinson
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

// C++ headers
#include <math.h>
#include <vector>

// Library headers

// Project headers
#include "rbf_ops.h"
#include "../shared_fht_functions/hadamard_transforms.h"
#include "../shared_fht_functions/shared_rfgen_ops.h"

namespace nb = nanobind;


namespace CPURBFKernelCalculations {



template <typename T>
int rbfFeatureGen_(nb::ndarray<T, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> input_arr,
nb::ndarray<double, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> output_arr,
nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cpu, nb::c_contig> radem,
nb::ndarray<T, nb::shape<-1>, nb::device::cpu, nb::c_contig> chi_arr,
bool fit_intercept) {
    // Perform safety checks. Any exceptions thrown here are handed
    // off to Python by the Nanobind wrapper. We do not expect the user
    // to see these because the Python code will always ensure inputs
    // are correct -- these are a failsafe -- so we do not need to provide
    // detailed exception messages here.
    int xDim0 = input_arr.shape(0);
    int xDim1 = input_arr.shape(1);
    size_t radem_shape2 = radem.shape(2);
    size_t num_rffs = output_arr.shape(1);
    size_t num_freqs = chi_arr.shape(0);
    double num_freqsFlt = num_freqs;

    T *input_ptr = static_cast<T*>(input_arr.data());
    double *output_ptr = static_cast<double*>(output_arr.data());
    T *chi_ptr = static_cast<T*>(chi_arr.data());
    int8_t *radem_ptr = static_cast<int8_t*>(radem.data());

    if (input_arr.shape(0) == 0 || output_arr.shape(0) != input_arr.shape(0))
        throw std::runtime_error("no datapoints");
    if (num_rffs < 2 || (num_rffs & 1) != 0)
        throw std::runtime_error("last dim of output must be even number");
    if ( (2 * num_freqs) != num_rffs || num_freqs > radem.shape(2) )
        throw std::runtime_error("incorrect number of rffs and or freqs.");

    double expectedNFreq = (xDim1 > 2) ? static_cast<double>(xDim1) : 2.0;
    double log2Freqs = std::log2(expectedNFreq);
    log2Freqs = std::ceil(log2Freqs);
    int padded_buffer_size = std::pow(2, log2Freqs);

    if (radem.shape(2) % padded_buffer_size != 0)
        throw std::runtime_error("incorrect number of rffs and or freqs.");

    T rbf_norm_constant;

    if (fit_intercept)
        rbf_norm_constant = std::sqrt(1.0 / (num_freqsFlt - 0.5));
    else
        rbf_norm_constant = std::sqrt(1.0 / num_freqsFlt);


    #pragma omp parallel
    {
    int repeat_position;
    int num_repeats = (num_freqs + padded_buffer_size - 1) / padded_buffer_size;
    // Notice that we don't have error handling here...very naughty. Out of
    // memory should be extremely rare since we are only allocating memory
    // for one row of the convolution. TODO: add error handling here.
    T *copy_buffer = new T[padded_buffer_size];

    #pragma omp for
    for (int i=0; i < xDim0; i++) {
        repeat_position = 0;

        for (int k=0; k < num_repeats; k++) {
            int start_pos = i * xDim1;
            #pragma omp simd
            for (int m=0; m < xDim1; m++)
                copy_buffer[m] = input_ptr[start_pos + m];
            #pragma omp simd
            for (int m=xDim1; m < padded_buffer_size; m++)
                copy_buffer[m] = 0;

            SharedCPURandomFeatureOps::singleVectorSORF(copy_buffer,
                    radem_ptr, repeat_position, radem_shape2, padded_buffer_size);
            SharedCPURandomFeatureOps::singleVectorRBFPostProcess(copy_buffer,
                    chi_ptr, output_ptr, padded_buffer_size, num_freqs, i,
                    k, rbf_norm_constant);
            repeat_position += padded_buffer_size;
        }
    }
    delete[] copy_buffer;
    }

    return 0;
}
//Explicitly instantiate so wrapper can use.
template int rbfFeatureGen_<double>(nb::ndarray<double, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> input_arr,
nb::ndarray<double, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> output_arr,
nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cpu, nb::c_contig> radem,
nb::ndarray<double, nb::shape<-1>, nb::device::cpu, nb::c_contig> chi_arr,
bool fit_intercept);

template int rbfFeatureGen_<float>(nb::ndarray<float, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> input_arr,
nb::ndarray<double, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> output_arr,
nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cpu, nb::c_contig> radem,
nb::ndarray<float, nb::shape<-1>, nb::device::cpu, nb::c_contig> chi_arr,
bool fit_intercept);




/// @brief Calculates both the random features and the gradient for the
/// RBF kernel.
/// @param input_arr The input features that will be used to generate the RFs and
/// gradient.
/// @param output_arr The array in which random features will be stored.
/// @param precompWeights The precomputed weight matrix for converting from
/// input to random features.
/// @param radem The array storing the diagonal Rademacher matrices.
/// @param chi_arr The array storing the diagonal scaling matrix.
/// @param sigma The lengthscale hyperparameter.
/// @param fit_intercept Whether to convert the first column to all 1s to fit
/// an intercept.
template <typename T>
int rbfGrad_(nb::ndarray<T, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> input_arr,
nb::ndarray<double, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> output_arr,
nb::ndarray<double, nb::shape<-1,-1,1>, nb::device::cpu, nb::c_contig> grad_arr,
nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cpu, nb::c_contig> radem,
nb::ndarray<T, nb::shape<-1>, nb::device::cpu, nb::c_contig> chi_arr,
double sigma, bool fit_intercept) {
    // Perform safety checks. Any exceptions thrown here are handed
    // off to Python by the Nanobind wrapper. We do not expect the user
    // to see these because the Python code will always ensure inputs
    // are correct -- these are a failsafe -- so we do not need to provide
    // detailed exception messages here.
    int xDim0 = input_arr.shape(0);
    int xDim1 = input_arr.shape(1);
    size_t radem_shape2 = radem.shape(2);
    size_t num_rffs = output_arr.shape(1);
    size_t num_freqs = chi_arr.shape(0);
    double num_freqsFlt = num_freqs;

    T *input_ptr = static_cast<T*>(input_arr.data());
    double *output_ptr = static_cast<double*>(output_arr.data());
    double *gradientPtr = static_cast<double*>(grad_arr.data());
    T *chi_ptr = static_cast<T*>(chi_arr.data());
    int8_t *radem_ptr = static_cast<int8_t*>(radem.data());

    if (input_arr.shape(0) == 0 ||
            output_arr.shape(0) != input_arr.shape(0))
        throw std::runtime_error("no datapoints");
    if (num_rffs < 2 || (num_rffs & 1) != 0)
        throw std::runtime_error("last dim of output must be even number");
    if ( (2 * num_freqs) != num_rffs || num_freqs > radem.shape(2) )
        throw std::runtime_error("incorrect number of rffs and or freqs.");
    if (grad_arr.shape(0) != output_arr.shape(0) ||
            grad_arr.shape(1) != output_arr.shape(1))
        throw std::runtime_error("Wrong array sizes.");

    double expectedNFreq = (xDim1 > 2) ? static_cast<double>(xDim1) : 2.0;
    double log2Freqs = std::log2(expectedNFreq);
    log2Freqs = std::ceil(log2Freqs);
    int padded_buffer_size = std::pow(2, log2Freqs);

    if (radem.shape(2) % padded_buffer_size != 0)
        throw std::runtime_error("incorrect number of rffs and or freqs.");

    double rbf_norm_constant;

    if (fit_intercept)
        rbf_norm_constant = std::sqrt(1.0 / (num_freqsFlt - 0.5));
    else
        rbf_norm_constant = std::sqrt(1.0 / num_freqsFlt);


    #pragma omp parallel
    {
    int repeat_position;
    int num_repeats = (num_freqs + padded_buffer_size - 1) / padded_buffer_size;
    // Notice that we don't have error handling here...very naughty. Out of
    // memory should be extremely rare since we are only allocating memory
    // for one row of the convolution. TODO: add error handling here.
    T *copy_buffer = new T[padded_buffer_size];

    #pragma omp for
    for (int i=0; i < xDim0; i++) {
        repeat_position = 0;

        for (int k=0; k < num_repeats; k++) {
            int start_pos = i * xDim1;
            #pragma omp simd
            for (int m=0; m < xDim1; m++)
                copy_buffer[m] = input_ptr[start_pos + m];
            #pragma omp simd
            for (int m=xDim1; m < padded_buffer_size; m++)
                copy_buffer[m] = 0;

            SharedCPURandomFeatureOps::singleVectorSORF(copy_buffer,
                    radem_ptr, repeat_position, radem_shape2, padded_buffer_size);
            SharedCPURandomFeatureOps::singleVectorRBFPostGrad(copy_buffer,
                    chi_ptr, output_ptr, gradientPtr, sigma, padded_buffer_size,
                    num_freqs, i, k, rbf_norm_constant);
            repeat_position += padded_buffer_size;
        }
    }
    delete[] copy_buffer;
    }

    return 0;
}
//Explicitly instantiate for external use.
template int rbfGrad_<double>(nb::ndarray<double, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> input_arr,
nb::ndarray<double, nb::shape<-1, -1>, nb::device::cpu, nb::c_contig> output_arr,
nb::ndarray<double, nb::shape<-1, -1, 1>, nb::device::cpu, nb::c_contig> grad_arr,
nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cpu, nb::c_contig> radem,
nb::ndarray<double, nb::shape<-1>, nb::device::cpu, nb::c_contig> chi_arr,
double sigma, bool fit_intercept);

template int rbfGrad_<float>(nb::ndarray<float, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> input_arr,
nb::ndarray<double, nb::shape<-1, -1>, nb::device::cpu, nb::c_contig> output_arr,
nb::ndarray<double, nb::shape<-1, -1, 1>, nb::device::cpu, nb::c_contig> grad_arr,
nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cpu, nb::c_contig> radem,
nb::ndarray<float, nb::shape<-1>, nb::device::cpu, nb::c_contig> chi_arr,
double sigma, bool fit_intercept);


}  // namespace CPURBFKernelCalculations
