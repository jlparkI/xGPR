/* Copyright (C) 2025 Jonathan Parkinson
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#ifndef CUDA_CLASSIFICATION_CALCS_HEADER_H_
#define CUDA_CLASSIFICATION_CALCS_HEADER_H_
// C++ headers
#include <stdint.h>

// Library headers
#include "nanobind/nanobind.h"
#include <nanobind/ndarray.h>

// Project headers

namespace nb = nanobind;

namespace CudaClassificationCalcs {

void cudaFindClassMeans_(nb::ndarray<double, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> input_arr,
    nb::ndarray<double, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> class_means,
    nb::ndarray<int32_t, nb::shape<-1>, nb::device::cuda, nb::c_contig> class_labels,
    nb::ndarray<int64_t, nb::shape<-1>, nb::device::cuda, nb::c_contig> class_counts);

void cudaPrepPooledCovCalc_(nb::ndarray<double, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> input_arr,
    nb::ndarray<double, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> class_means,
    nb::ndarray<int32_t, nb::shape<-1>, nb::device::cuda, nb::c_contig> class_labels,
    nb::ndarray<double, nb::shape<-1>, nb::device::cuda, nb::c_contig> class_priors);

}  // namespace CudaClassificationCalcs


#endif  // CUDA_CLASSIFICATION_CALCS_H
