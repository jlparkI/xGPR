/* Copyright (C) 2025 Jonathan Parkinson
*/
#ifndef CONV1D_OPERATIONS_H
#define CONV1D_OPERATIONS_H

// C++ headers
#include <stdint.h>

// Library headers
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

// Project headers


namespace nb = nanobind;




namespace CPUMaxpoolKernelCalculations {


/// @brief Generates random features for the static layer FastConv1d-type
/// feature generator.
/// @param input_arr The input features that will be used to generate the RFs.
/// @param output_arr The array in which random features will be stored.
/// @param radem The array in which the diagonal Rademacher matrices are
/// stored.
/// @param chi_arr The array in which the diagonal scaling matrix is stored.
/// @param seqlengths The array storing the length of each sequence in
/// input_arr.
/// @param conv_width The width of the convolution kernel.
template <typename T>
int conv1dMaxpoolFeatureGen_(
nb::ndarray<T, nb::shape<-1,-1,-1>, nb::device::cpu, nb::c_contig> input_arr,
nb::ndarray<float, nb::shape<-1, -1>, nb::device::cpu, nb::c_contig> output_arr,
nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cpu, nb::c_contig> radem,
nb::ndarray<T, nb::shape<-1>, nb::device::cpu, nb::c_contig> chi_arr,
nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> seqlengths,
int conv_width);



/// @brief Performs the post-Hadamard transform operations
/// needed to generate the kernel output on a single vector.
/// @param xdata The vector to operate on.
/// @param chi_arr The array in which the diagonal scaling matrix is stored.
/// @param output_array The pointer to the first element of the 2d
/// array in which results will be stored.
/// @param dim2 dim2 of output_array.
/// @param num_freqs The number of frequencies that are sampled to
/// approximate the kernel. This is the size of xdata.
/// @param row_number The row of output_array where the output should
/// be stored.
/// @param repeat_num Depending on num_freqs, we may need to transform
/// each row of input multiple times to generate one row of output.
/// This is the number of repeats we have performed so far.
template <typename T>
void singleVectorMaxpoolPostProcess(const T xdata[],
const T chi_arr[], float *output_array,
int dim2, int num_freqs,
int row_number, int repeat_num);


}  // namespace CPUMaxpoolKernelCalculations


#endif
