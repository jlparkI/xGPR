#ifndef CUDA_RBF_SPECIFIC_CONVOLUTION_H
#define CUDA_RBF_SPECIFIC_CONVOLUTION_H

#include <stdint.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;




template <typename T>
int convRBFFeatureGen(
nb::ndarray<T, nb::shape<-1,-1,-1>, nb::device::cuda, nb::c_contig> input_arr,
nb::ndarray<double, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> output_arr,
nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cuda, nb::c_contig> radem,
nb::ndarray<T, nb::shape<-1>, nb::device::cuda, nb::c_contig> chi_arr,
nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> seqlengths,
int conv_width, int scaling_type);

template <typename T>
int convRBFFeatureGrad(
nb::ndarray<T, nb::shape<-1,-1,-1>, nb::device::cuda, nb::c_contig> input_arr,
nb::ndarray<double, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> output_arr,
nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cuda, nb::c_contig> radem,
nb::ndarray<T, nb::shape<-1>, nb::device::cuda, nb::c_contig> chi_arr,
nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> seqlengths,
nb::ndarray<double, nb::shape<-1,-1,1>, nb::device::cuda, nb::c_contig> grad_arr,
double sigma, int conv_width, int scaling_type);

#endif
