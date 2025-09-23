/* Copyright (C) 2025 Jonathan Parkinson
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#ifndef SPEC_CPU_RBF_OPS_H
#define SPEC_CPU_RBF_OPS_H

// C++ headers
#include <stdint.h>

// Library headers
#include "nanobind/nanobind.h"
#include <nanobind/ndarray.h>

// Project headers

namespace nb = nanobind;


namespace CPURBFKernelCalculations {


/// @brief Calculates the random features only for the RBF kernel.
/// @param input_arr The input features that will be used to generate the RFs.
/// @param output_arr The array in which random features will be stored.
/// @param radem The array storing the diagonal Rademacher matrices.
/// @param chi_arr The array storing the diagonal scaling matrix.
/// @param fit_intercept Whether to convert the first column to all 1s to fit
/// an intercept.
template <typename T>
int rbfFeatureGen_(nb::ndarray<T, nb::shape<-1, -1>, nb::device::cpu, nb::c_contig> input_arr,
nb::ndarray<double, nb::shape<-1, -1>, nb::device::cpu, nb::c_contig> output_arr,
nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cpu, nb::c_contig> radem,
nb::ndarray<T, nb::shape<-1>, nb::device::cpu, nb::c_contig> chi_arr,
bool fit_intercept);


/// @brief Calculates both the random features and the gradient for the
/// RBF kernel.
/// @param input_arr The input features that will be used to generate the RFs and
/// gradient.
/// @param output_arr The array in which random features will be stored.
/// @param grad_arr The array in which the gradient will be stored.
/// @param radem The array storing the diagonal Rademacher matrices.
/// @param chi_arr The array storing the diagonal scaling matrix.
/// @param sigma The lengthscale hyperparameter.
/// @param fit_intercept Whether to convert the first column to all 1s to fit
/// an intercept.
template <typename T>
int rbfGrad_(nb::ndarray<T, nb::shape<-1, -1>, nb::device::cpu, nb::c_contig> input_arr,
nb::ndarray<double, nb::shape<-1, -1>, nb::device::cpu, nb::c_contig> output_arr,
nb::ndarray<double, nb::shape<-1, -1, 1>, nb::device::cpu, nb::c_contig> grad_arr,
nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cpu, nb::c_contig> radem,
nb::ndarray<T, nb::shape<-1>, nb::device::cpu, nb::c_contig> chi_arr,
double sigma, bool fit_intercept);


}  // namespace CPURBFKernelCalculations

#endif
