/* Copyright (C) 2025 Jonathan Parkinson
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#ifndef SHARED_RFGEN_OPERATIONS_H
#define SHARED_RFGEN_OPERATIONS_H
// C++ headers
#include <stdint.h>

// Library headers

// Project headers


namespace SharedCPURandomFeatureOps {

template <typename T>
void multiplyByDiagonalRademacherMat2D(
T __restrict xArray[],
const int8_t *rademArray,
int dim1,
int startRow, int endRow);

template <typename T>
void multiplyByDiagonalRademacherMat(
T __restrict xArray[],
const int8_t *rademArray,
int dim1, int dim2,
int startRow, int endRow);


/// @brief Runs the SORF operation on a single input vector.
/// @param cbuffer The input vector on which to run SORF.
/// @param rademArray The diagonal rademacher matrix.
/// @param repeatPosition The number of repeats of SORF that
/// have occurred so far (indicates starting position in
/// rademArray).
/// @param rademShape2 rademArray has shape (3,1,rademShape2).
/// @param cbufferDim2 The length of cbuffer.
template <typename T>
void singleVectorSORF(
T cbuffer[], const int8_t *rademArray,
int repeatPosition, int rademShape2,
int cbufferDim2);


/// @brief Runs the final RBF feature generation calculations
/// after SORF.
/// @param xdata The input vector to which SORF has been
/// applied.
/// @param chiArr A diagonal matrix drawn from the chi
/// distribution.
/// @param outputArray The array in which the results will
/// be stored.
/// @param dim2 The shape of xdata.
/// @param numFreqs The size of chiArr; also the number of
/// random frequencies that are sampled.
/// @param rowNumber The row of outputArray in which to
/// store results.
/// @param repeatNum The number of SORF operations already
/// completed. Indicates where to start in chiArr.
/// @param scalingTerm A normalization constant determined
/// by caller.
template <typename T>
void singleVectorRBFPostProcess(
const T xdata[],
const T chiArr[], double *outputArray,
int dim2, int numFreqs,
int rowNumber, int repeatNum,
double scalingTerm);

template <typename T>
void singleVectorRBFPostGrad(
const T xdata[],
const T chiArr[], double *outputArray,
double *gradientArray, double sigma,
int dim2, int numFreqs,
int rowNumber, int repeatNum,
double scalingTerm);

}  // namespace SharedCPURandomFeatureOps

#endif
