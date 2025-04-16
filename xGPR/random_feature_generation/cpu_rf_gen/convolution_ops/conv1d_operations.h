/* Copyright (C) 2025 Jonathan Parkinson
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#ifndef CONV1D_OPERATIONS_H
#define CONV1D_OPERATIONS_H

// C++ headers
#include <stdint.h>

// Library headers
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

// Project headers


namespace nb = nanobind;




namespace CPUMaxpoolKernelCalculations {

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))


/// @brief Generates random features for the static layer FastConv1d-type
/// feature generator.
/// @param inputArr The input features that will be used to generate the RFs.
/// @param outputArr The array in which random features will be stored.
/// @param radem The array in which the diagonal Rademacher matrices are
/// stored.
/// @param chiArr The array in which the diagonal scaling matrix is stored.
/// @param seqlengths The array storing the length of each sequence in
/// inputArr.
/// @param convWidth The width of the convolution kernel.
template <typename T>
int conv1dMaxpoolFeatureGen_(
nb::ndarray<T, nb::shape<-1,-1,-1>, nb::device::cpu, nb::c_contig> inputArr,
nb::ndarray<float, nb::shape<-1, -1>, nb::device::cpu, nb::c_contig> outputArr,
nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cpu, nb::c_contig> radem,
nb::ndarray<T, nb::shape<-1>, nb::device::cpu, nb::c_contig> chiArr,
nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> seqlengths,
int convWidth);



template <typename T>
void singleVectorMaxpoolPostProcess(const T xdata[],
const T chiArr[], float *outputArray,
int dim2, int numFreqs,
int rowNumber, int repeatNum);


}  // namespace CPUMaxpoolKernelCalculations


#endif
