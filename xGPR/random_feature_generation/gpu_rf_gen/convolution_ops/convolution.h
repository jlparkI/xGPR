/* Copyright (C) 2025 Jonathan Parkinson
*/
#ifndef CUDA_CONVOLUTION_H
#define CUDA_CONVOLUTION_H
// C++ headers
#include <stdint.h>

// Library headers
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

// Project headers

namespace nb = nanobind;



template <typename T>
int conv1dMaxpoolFeatureGen(
nb::ndarray<T, nb::shape<-1,-1,-1>, nb::device::cuda, nb::c_contig> input_arr,
nb::ndarray<float, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> output_arr,
nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cuda, nb::c_contig> radem,
nb::ndarray<T, nb::shape<-1>, nb::device::cuda, nb::c_contig> chi_arr,
nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> seqlengths,
int conv_width);



#endif
