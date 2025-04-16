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
/// @param inputArr The input features that will be used to generate the RFs.
/// @param outputArr The array in which random features will be stored.
/// @param radem The array storing the diagonal Rademacher matrices.
/// @param chiArr The array storing the diagonal scaling matrix.
/// @param fitIntercept Whether to convert the first column to all 1s to fit
/// an intercept.
template <typename T>
int rbfFeatureGen_(nb::ndarray<T, nb::shape<-1, -1>, nb::device::cpu, nb::c_contig> inputArr,
nb::ndarray<double, nb::shape<-1, -1>, nb::device::cpu, nb::c_contig> outputArr,
nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cpu, nb::c_contig> radem,
nb::ndarray<T, nb::shape<-1>, nb::device::cpu, nb::c_contig> chiArr,
bool fitIntercept);


/// @brief Calculates both the random features and the gradient for the
/// RBF kernel.
/// @param inputArr The input features that will be used to generate the RFs and
/// gradient.
/// @param outputArr The array in which random features will be stored.
/// @param precompWeights The precomputed weight matrix for converting from
/// input to random features.
/// @param radem The array storing the diagonal Rademacher matrices.
/// @param chiArr The array storing the diagonal scaling matrix.
/// @param sigma The lengthscale hyperparameter.
/// @param fitIntercept Whether to convert the first column to all 1s to fit
/// an intercept.
template <typename T>
int rbfGrad_(nb::ndarray<T, nb::shape<-1, -1>, nb::device::cpu, nb::c_contig> inputArr,
nb::ndarray<double, nb::shape<-1, -1>, nb::device::cpu, nb::c_contig> outputArr,
nb::ndarray<double, nb::shape<-1, -1, 1>, nb::device::cpu, nb::c_contig> gradArr,
nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cpu, nb::c_contig> radem,
nb::ndarray<T, nb::shape<-1>, nb::device::cpu, nb::c_contig> chiArr,
double sigma, bool fitIntercept);


}  // namespace CPURBFKernelCalculations

#endif
