/* Copyright (C) 2025 Jonathan Parkinson
*/
#ifndef CUDA_RBF_SPEC_OPERATIONS_H
#define CUDA_RBF_SPEC_OPERATIONS_H

#include <stdint.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;


template <typename T>
int RBFFeatureGen(
nb::ndarray<const T, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> input_arr,
nb::ndarray<double, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> output_arr,
nb::ndarray<const int8_t, nb::shape<3,1,-1>, nb::device::cuda, nb::c_contig> radem,
nb::ndarray<const T, nb::shape<-1>, nb::device::cuda, nb::c_contig> chi_arr,
bool fit_intercept);

template <typename T>
int RBFFeatureGrad(
nb::ndarray<const T, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> input_arr,
nb::ndarray<double, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> output_arr,
nb::ndarray<double, nb::shape<-1,-1,1>, nb::device::cuda, nb::c_contig> grad_arr,
nb::ndarray<const int8_t, nb::shape<3,1,-1>, nb::device::cuda, nb::c_contig> radem,
nb::ndarray<const T, nb::shape<-1>, nb::device::cuda, nb::c_contig> chi_arr,
float sigma, bool fit_intercept);

#endif
