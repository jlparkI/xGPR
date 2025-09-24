/* Copyright (C) 2025 Jonathan Parkinson
*/
#ifndef RBF_CONVOLUTION_H
#define RBF_CONVOLUTION_H

// C++ headers
#include <stdint.h>

// Library headers
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

// Project headers

namespace nb = nanobind;


namespace CPURBFConvolutionKernelCalculations {


static constexpr int NO_CONVOLUTION_SCALING = 0;
static constexpr int SQRT_CONVOLUTION_SCALING = 1;
static constexpr int FULL_CONVOLUTION_SCALING = 2;



/// @brief Generates random features for RBF-based convolution kernels.
/// @param input_arr The input features that will be used to generate the RFs.
/// @param output_arr The array in which random features will be stored.
/// @param radem The array in which the diagonal Rademacher matrices are
/// stored.
/// @param chi_arr The array in which the diagonal scaling matrix is stored.
/// @param seqlengths The array storing the length of each sequence in
/// input_arr.
/// @param conv_width The width of the convolution kernel.
/// @param scaling_type The type of scaling to perform (i.e. how to normalize
/// for different sequence lengths).
template <typename T>
int convRBFFeatureGen_(nb::ndarray<T, nb::shape<-1,-1,-1>, nb::device::cpu, nb::c_contig> input_arr,
nb::ndarray<double, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> output_arr,
nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cpu, nb::c_contig> radem,
nb::ndarray<T, nb::shape<-1>, nb::device::cpu, nb::c_contig> chi_arr,
nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> seqlengths,
int conv_width, int scaling_type);



/// @brief Generates both random features and gradient for
/// RBF-based convolution kernels.
/// @param input_arr The input features that will be used to generate the RFs.
/// @param output_arr The array in which random features will be stored.
/// @param radem The array in which the diagonal Rademacher matrices are
/// stored.
/// @param chi_arr The array in which the diagonal scaling matrix is stored.
/// @param seqlengths The array storing the length of each sequence in
/// input_arr.
/// @param grad_arr The array in which the gradient will be stored.
/// @param conv_width The width of the convolution kernel.
/// @param sigma The lengthscale hyperparameter.
/// @param scaling_type The type of scaling to perform (i.e. how to normalize
/// for different sequence lengths).
template <typename T>
int convRBFGrad_(nb::ndarray<T, nb::shape<-1,-1,-1>, nb::device::cpu, nb::c_contig> input_arr,
nb::ndarray<double, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> output_arr,
nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cpu, nb::c_contig> radem,
nb::ndarray<T, nb::shape<-1>, nb::device::cpu, nb::c_contig> chi_arr,
nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> seqlengths,
nb::ndarray<double, nb::shape<-1,-1,1>, nb::device::cpu, nb::c_contig> grad_arr,
double sigma, int conv_width, int scaling_type);


}  // namespace CPURBFConvolutionKernelCalculations


#endif
