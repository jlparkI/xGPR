/* Copyright (C) 2025 Jonathan Parkinson
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



/// @brief Performs a fast Hadamard transform along the last
/// dimension of a 3d array.
/// @param input_arr The array on which the transform will
/// be performed in place. Last dim must be a power of 2.
/// @return An error code; 0 if no error.
template <typename T>
int fastHadamard3dArray_(
nb::ndarray<T, nb::shape<-1, -1, -1>,
nb::device::cpu, nb::c_contig> inputArr);



/// @brief Performs a fast Hadamard transform along the last
/// dimension of a 2d array.
/// @param input_arr The array on which the transform will
/// be performed in place.
/// @return An error code; 0 if no error.
template <typename T>
int fastHadamard2dArray_(
nb::ndarray<T, nb::shape<-1, -1>,
nb::device::cpu, nb::c_contig> inputArr);



/// @brief Performs the H D portion of an SRHT operation
/// in place on the 2d input array.
/// @param input_arr The array on which the transform will
/// be performed in place. Last dim must be a power of 2.
/// @param radem A diagonal int8 1d array whose size is the
/// same as dim2 of input_arr.
/// @return An error code; 0 if no error.
template <typename T>
int SRHTBlockTransform(
nb::ndarray<T, nb::shape<-1, -1>, nb::device::cpu, nb::c_contig> inputArr,
nb::ndarray<int8_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> radem);




}  // namespace CPUHadamardTransformBasicCalculations


#endif
