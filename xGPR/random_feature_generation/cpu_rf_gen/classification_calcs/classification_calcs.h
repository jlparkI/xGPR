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

/// @brief Adds to a rolling tally to compute class means and the number of
/// datapoints in each class.
/// @param input_arr The features for a block of datapoints.
/// @param class_means The array in which the sum of all datapoints for
/// each class to date is stored.
/// @param class_labels The class labels for a block of datapoints.
/// @param class_counts The number of datapoints in each class to date.
void cpuFindClassMeans_(nb::ndarray<double, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> input_arr,
        nb::ndarray<double, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> class_means,
        nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> class_labels,
        nb::ndarray<int64_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> class_counts);


/// @brief Preps the features for a block of datapoints for the
/// covariance matrix calculation.
/// @param input_arr The features for a block of datapoints.
/// @param class_means The mean of all datapoints in a given class for
/// each class.
/// @param class_labels The class labels for a block of datapoints.
/// @param class_counts The prior probability of each class (generally
/// estimated from the data, but could also be set by user).
void cpuPrepPooledCovCalc_(nb::ndarray<double, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> input_arr,
        nb::ndarray<double, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> class_means,
        nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> class_labels,
        nb::ndarray<double, nb::shape<-1>, nb::device::cpu, nb::c_contig> class_prior_sqrts);

}  // namespace CpuClassificationCalcs


#endif  // CPU_CLASSIFICATION_CALCS_H
