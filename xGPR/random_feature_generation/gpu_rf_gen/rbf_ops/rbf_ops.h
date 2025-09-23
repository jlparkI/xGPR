#ifndef CUDA_RBF_SPEC_OPERATIONS_H
#define CUDA_RBF_SPEC_OPERATIONS_H

#include <stdint.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;



template <typename T>
int RBFFeatureGen(
        nb::ndarray<const T, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> inputArr,
        nb::ndarray<double, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> outputArr,
        nb::ndarray<const int8_t, nb::shape<3,1,-1>, nb::device::cuda, nb::c_contig> radem,
        nb::ndarray<const T, nb::shape<-1>, nb::device::cuda, nb::c_contig> chiArr,
        bool fitIntercept);

template <typename T>
int RBFFeatureGrad(
        nb::ndarray<const T, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> inputArr,
        nb::ndarray<double, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> outputArr,
        nb::ndarray<double, nb::shape<-1,-1,1>, nb::device::cuda, nb::c_contig> gradArr,
        nb::ndarray<const int8_t, nb::shape<3,1,-1>, nb::device::cuda, nb::c_contig> radem,
        nb::ndarray<const T, nb::shape<-1>, nb::device::cuda, nb::c_contig> chiArr,
        float sigma, bool fitIntercept);


#endif
