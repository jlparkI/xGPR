/* Copyright (C) 2025 Jonathan Parkinson
*/
#ifndef SHARED_RFGEN_OPERATIONS_H
#define SHARED_RFGEN_OPERATIONS_H
// C++ headers
#include <stdint.h>

// Library headers

// Project headers


namespace SharedCPURandomFeatureOps {

/// @brief Multiplies a 2d array by a diagonal matrix.
/// @param xArray Pointer to the first element of the array
/// to transform (in place).
/// @param radem_array The diagonal rademacher matrix.
/// @param dim1 The second dimension of xArray. radem_array
/// must also be of this size.
/// @param start_row The first row of xArray to transform.
/// @param end_row The last row of xArray to transform.
template <typename T>
void multiplyByDiagonalRademacherMat2D(
T __restrict xArray[],
const int8_t *radem_array,
int dim1,
int start_row, int end_row);


/// @brief Runs the SORF operation on a single input vector.
/// @param cbuffer The input vector on which to run SORF.
/// @param radem_array The diagonal rademacher matrix.
/// @param repeatPosition The number of repeats of SORF that
/// have occurred so far (indicates starting position in
/// radem_array).
/// @param rademShape2 radem_array has shape (3,1,rademShape2).
/// @param cbufferDim2 The length of cbuffer.
template <typename T>
void singleVectorSORF(
T cbuffer[], const int8_t *radem_array,
int repeat_position, int rademShape2,
int cbufferDim2);


/// @brief Runs the final RBF feature generation calculations
/// after SORF.
/// @param xdata The input vector to which SORF has been
/// applied.
/// @param chi_arr A diagonal matrix drawn from the chi
/// distribution.
/// @param output_array The array in which the results will
/// be stored.
/// @param dim2 The shape of xdata.
/// @param num_freqs The size of chi_arr; also the number of
/// random frequencies that are sampled.
/// @param row_number The row of output_array in which to
/// store results.
/// @param repeat_num The number of SORF operations already
/// completed. Indicates where to start in chi_arr.
/// @param scaling_term A normalization constant determined
/// by caller.
template <typename T>
void singleVectorRBFPostProcess(
const T xdata[],
const T chi_arr[], double *output_array,
int dim2, int num_freqs,
int row_number, int repeat_num,
double scaling_term);


/// @brief Runs the final RBF feature generation calculations
/// after SORF, also calculating a gradient in the process.
/// @param xdata The input vector to which SORF has been
/// applied.
/// @param chi_arr A diagonal matrix drawn from the chi
/// distribution.
/// @param output_array The array in which the results will
/// be stored.
/// @param gradient_array The array in which the gradient is
/// stored.
/// @param sigma The lengthscale hyperparameter for RBF.
/// @param dim2 The shape of xdata.
/// @param num_freqs The size of chi_arr; also the number of
/// random frequencies that are sampled.
/// @param row_number The row of output_array in which to
/// store results.
/// @param repeat_num The number of SORF operations already
/// completed. Indicates where to start in chi_arr.
/// @param scaling_term A normalization constant determined
/// by caller.
template <typename T>
void singleVectorRBFPostGrad(
const T xdata[],
const T chi_arr[], double *output_array,
double *gradient_array, double sigma,
int dim2, int num_freqs,
int row_number, int repeat_num,
double scaling_term);

}  // namespace SharedCPURandomFeatureOps

#endif
