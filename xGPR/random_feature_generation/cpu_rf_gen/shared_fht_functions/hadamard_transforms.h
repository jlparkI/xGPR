/* Copyright (C) 2025 Jonathan Parkinson
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#ifndef HADAMARD_TRANSFORM_OPERATIONS_H
#define HADAMARD_TRANSFORM_OPERATIONS_H

#define MIN(a, b) ((a) < (b) ? (a) : (b))

namespace CPUHadamardTransformOps {

template <typename T>
void transformRows(T __restrict__ xArray[], int startRow, int endRow,
                    int dim1, int dim2);


template <typename T>
void singleVectorTransform(T __restrict__ xArray[], int dim);


}  // namespace CPUHadamardTransformOps

#endif
