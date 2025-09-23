/* Copyright (C) 2025 Jonathan Parkinson
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
// C++ headers
#include <math.h>
#include <cstring>

// Library headers

// Project headers
#include "hadamard_transforms.h"


namespace CPUHadamardTransformOps {




template <typename T>
void transformRows(T __restrict__ xArray[], int startRow, int endRow,
                    int dim1, int dim2) {
    T y;
    int rowStride = dim1 * dim2;
    int end_pos = endRow * rowStride;

    for (int idx1 = startRow; idx1 < endRow; idx1++) {
        int start_pos = idx1 * rowStride;
        int end_pos = start_pos + rowStride;

        #pragma omp simd
        for (int i = start_pos; i < end_pos; i += 2) {
            y = xArray[i+1];
            xArray[i+1] = xArray[i] - y;
            xArray[i] += y;
        }
        if (dim2 <= 2)
            continue;


        for (int i = start_pos; i < end_pos; i += 4) {
            #pragma omp simd
            for (int j=i; j < (i+2); j++) {
                y = xArray[j+2];
                xArray[j+2] = xArray[j] - y;
                xArray[j] += y;
            }
        }
        if (dim2 <= 4)
            continue;

        for (int i = start_pos; i < end_pos; i += 8) {
            #pragma omp simd
            for (int j=i; j < (i+4); j++) {
                y = xArray[j+4];
                xArray[j+4] = xArray[j] - y;
                xArray[j] += y;
            }
        }
        if (dim2 <= 8)
            continue;


        for (int h = 8; h < dim2; h <<= 1) {
            for (int i = start_pos; i < end_pos; i += (h << 1)) {
                #pragma omp simd
                for (int j=i; j < (i+h); j++) {
                    y = xArray[j+h];
                    xArray[j+h] = xArray[j] - y;
                    xArray[j] += y;
                }
            }
        }
    }
}
template void transformRows<double>(double *__restrict__ xArray,
int startRow, int endRow, int dim1, int dim2);
template void transformRows<float>(float *__restrict__ xArray,
int startRow, int endRow, int dim1, int dim2);






/*!
 * # singleVectorTransform
 *
 * Performs an unnormalized Hadamard transform along a single
 * vector, which allows for some simplifications.
 *
 * ## Args:
 *
 * + `xArray` Pointer to the first element of the array to be
 * modified. Must be a 1d array (e.g. C). C MUST be
 * a power of 2.
 * + `dim` The length of the array.
 */
template <typename T>
void singleVectorTransform(T __restrict__ xArray[], int dim) {
    T y;
    #pragma omp simd
    for (int i = 0; i < dim; i += 2) {
        y = xArray[i+1];
        xArray[i+1] = xArray[i] - y;
        xArray[i] += y;
    }
    if (dim <= 2)
        return;

    for (int i = 0; i < dim; i += 4) {
        #pragma omp simd
        for (int j=i; j < (i+2); j++) {
            y = xArray[j+2];
            xArray[j+2] = xArray[j] - y;
            xArray[j] += y;
        }
    }
    if (dim <= 4)
        return;

    for (int i = 0; i < dim; i += 8) {
        #pragma omp simd
        for (int j=i; j < (i+4); j++) {
            y = xArray[j+4];
            xArray[j+4] = xArray[j] - y;
            xArray[j] += y;
        }
    }
    if (dim <= 8)
        return;


    for (int h = 8; h < dim; h <<= 1) {
        for (int i = 0; i < dim; i += (h << 1)) {
            #pragma omp simd
            for (int j=i; j < (i+h); j++) {
                y = xArray[j+h];
                xArray[j+h] = xArray[j] - y;
                xArray[j] += y;
            }
        }
    }
}
template void singleVectorTransform<double>(double *__restrict__ xArray, int dim);
template void singleVectorTransform<float>(float *__restrict__ xArray, int dim);

}  // namespace CPUHadamardTransformOps
