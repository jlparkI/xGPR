#ifndef CUDA_CONVOLUTION_H
#define CUDA_CONVOLUTION_H

#include <stdint.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;



template <typename T>
int conv1dMaxpoolFeatureGen(nb::ndarray<T, nb::shape<-1,-1,-1>, nb::device::cuda, nb::c_contig> inputArr,
        nb::ndarray<float, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> outputArr,
        nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cuda, nb::c_contig> radem,
        nb::ndarray<T, nb::shape<-1>, nb::device::cuda, nb::c_contig> chiArr,
        nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> seqlengths,
        int convWidth);

#endif
