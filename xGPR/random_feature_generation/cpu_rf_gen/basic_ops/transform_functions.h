/* Copyright (C) 2025 Jonathan Parkinson
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#ifndef TRANSFORM_FUNCTIONS_H
#define TRANSFORM_FUNCTIONS_H

// C++ headers
#include <stdint.h>

// Library headers
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

// Project headers

namespace nb = nanobind;


namespace CPUHadamardTransformBasicCalculations {



template <typename T>
int fastHadamard3dArray_(
nb::ndarray<T, nb::shape<-1, -1, -1>,
nb::device::cpu, nb::c_contig> inputArr);



template <typename T>
int fastHadamard2dArray_(
nb::ndarray<T, nb::shape<-1, -1>,
nb::device::cpu, nb::c_contig> inputArr);



template <typename T>
int SRHTBlockTransform(
nb::ndarray<T, nb::shape<-1, -1>, nb::device::cpu, nb::c_contig> inputArr,
nb::ndarray<int8_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> radem);




}  // namespace CPUHadamardTransformBasicCalculations


#endif
