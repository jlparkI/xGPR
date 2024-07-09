#ifndef CUDA_ARD_SPEC_OPERATIONS_H
#define CUDA_ARD_SPEC_OPERATIONS_H

#include <stdint.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;



template <typename T>
int ardCudaGrad(nb::ndarray<T, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> inputArr,
        nb::ndarray<double, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> outputArr,
        nb::ndarray<T, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> precompWeights,
        nb::ndarray<int32_t, nb::shape<-1>, nb::device::cuda, nb::c_contig> sigmaMap,
        nb::ndarray<double, nb::shape<-1>, nb::device::cuda, nb::c_contig> sigmaVals,
        nb::ndarray<double, nb::shape<-1,-1,-1>, nb::device::cuda, nb::c_contig> gradArr,
        bool fitIntercept);

#endif
