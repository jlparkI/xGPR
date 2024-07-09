#ifndef BASIC_CUDA_FHT_ARRAY_OPERATIONS_H
#define BASIC_CUDA_FHT_ARRAY_OPERATIONS_H

#include <stdint.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;


template <typename T>
int cudaHTransform(nb::ndarray<T, nb::shape<-1,-1>, nb::device::cuda,
        nb::c_contig> inputArr);

template <typename T>
int cudaSRHT2d(nb::ndarray<T, nb::shape<-1,-1>, nb::device::cuda,
        nb::c_contig> inputArr,
        nb::ndarray<const int8_t, nb::shape<-1>, nb::device::cuda,
        nb::c_contig> radem,
        int numThreads);


#endif
