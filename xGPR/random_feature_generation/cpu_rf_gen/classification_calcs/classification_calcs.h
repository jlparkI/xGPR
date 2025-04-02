/* Copyright (C) 2025 Jonathan Parkinson
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#ifndef CPU_CLASSIFICATION_CALCS_HEADER_H_
#define CPU_CLASSIFICATION_CALCS_HEADER_H_
// C++ headers
#include <stdint.h>

// Library headers
#include "nanobind/nanobind.h"
#include <nanobind/ndarray.h>

// Project headers

namespace nb = nanobind;

namespace CpuClassificationCalcs {

void cpuFindClassMeans_(nb::ndarray<double, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> input_arr,
        nb::ndarray<double, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> class_means,
        nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> class_labels,
        nb::ndarray<int64_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> class_counts);

void cpuPrepPooledCovCalc_(nb::ndarray<double, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> input_arr,
        nb::ndarray<double, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> class_means,
        nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> class_labels,
        nb::ndarray<double, nb::shape<-1>, nb::device::cpu, nb::c_contig> class_priors);

}  // namespace CpuClassificationCalcs


#endif  // CPU_CLASSIFICATION_CALCS_H
