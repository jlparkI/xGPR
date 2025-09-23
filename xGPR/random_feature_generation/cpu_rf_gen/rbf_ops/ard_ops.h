/* Copyright (C) 2025 Jonathan Parkinson
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#ifndef SPEC_CPU_ARD_OPS_H
#define SPEC_CPU_ARD_OPS_H

// C++ headers
#include <stdint.h>

// Library headers
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

// Project headers



namespace nb = nanobind;


namespace CPUARDKernelCalculations {

/// @brief Calculates both the random features and the gradient for the MiniARD
/// kernel. TODO: This function needs further optimization.
/// @param inputArr The input features that will be used to generate the RFs and
/// gradient.
/// @param outputArr The array in which random features will be stored.
/// @param precompWeights The precomputed weight matrix for converting from
/// input to random features.
/// @param sigmaMap A map of which sigma value is applicable for
/// which input values.
/// @param sigmaVals The actual feature or region specific sigma hyperparameters
/// (lengthscales).
/// @param gradArr The array in which the calculated gradient will be stored.
/// @param fitIntercept Whether to convert the first column to all 1s to fit an
/// intercept.
template <typename T>
int ardGrad_(nb::ndarray<T, nb::shape<-1, -1>, nb::device::cpu, nb::c_contig> inputArr,
nb::ndarray<double, nb::shape<-1, -1>, nb::device::cpu, nb::c_contig> outputArr,
nb::ndarray<T, nb::shape<-1, -1>, nb::device::cpu, nb::c_contig> precompWeights,
nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> sigmaMap,
nb::ndarray<double, nb::shape<-1>, nb::device::cpu, nb::c_contig> sigmaVals,
nb::ndarray<double, nb::shape<-1, -1, -1>, nb::device::cpu, nb::c_contig> gradArr,
bool fitIntercept);


}  // namespace CPUARDKernelCalculations

#endif
