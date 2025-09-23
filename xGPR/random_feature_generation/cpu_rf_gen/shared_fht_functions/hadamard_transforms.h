/* Copyright (C) 2025 Jonathan Parkinson
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#ifndef HADAMARD_TRANSFORM_OPERATIONS_H
#define HADAMARD_TRANSFORM_OPERATIONS_H


namespace CPUHadamardTransformOps {


/// @brief Runs a fast Hadamard transform on the last dim of a 3d
/// array.
/// @param xArray pointer to the first element of the array to be
/// transformed in place.
/// @param start_row The first row of the array to transform.
/// @param end_row The last row of the array to transform.
/// @param dim1 dim1 (the second of the three dimensions of the
/// array).
/// @param dim2 dim2 (the third of the three dimensions of the
/// array).
template <typename T>
void transformRows(T __restrict__ xArray[], int start_row, int end_row,
                    int dim1, int dim2);


/// @brief Runs the fast Hadamard transform on an input vector.
/// @param xArray The vector to transform.
/// @param dim The number of elements in xArray. Must be a power
/// of 2.
template <typename T>
void singleVectorTransform(T __restrict__ xArray[], int dim);


}  // namespace CPUHadamardTransformOps

#endif
