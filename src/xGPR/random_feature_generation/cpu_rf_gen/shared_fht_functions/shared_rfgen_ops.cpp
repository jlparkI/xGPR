/* Copyright (C) 2025 Jonathan Parkinson
*/
// C++ headers
#include <math.h>
#include <cstring>
#include <algorithm>

// Library headers

// Project headers
#include "shared_rfgen_ops.h"
#include "hadamard_transforms.h"


namespace SharedCPURandomFeatureOps {


template <typename T>
void multiplyByDiagonalRademacherMat2D(T __restrict xArray[],
const int8_t *radem_array,
int dim1,
int start_row, int end_row) {
    T norm_constant = log2(dim1) / 2;
    norm_constant = 1 / pow(2, norm_constant);
    int row_stride = dim1;
    T *__restrict xElement;

    for (int i = start_row; i < end_row; i++) {
        xElement = xArray + i * row_stride;
        #pragma omp simd
        for (int j = 0; j < row_stride; j++)
            xElement[j] *= radem_array[j] * norm_constant;
    }
}
//Explicitly instantiate for external use.
template void multiplyByDiagonalRademacherMat2D<float>(float *__restrict xArray,
const int8_t *radem_array,
int dim1,
int start_row, int end_row);
template void multiplyByDiagonalRademacherMat2D<double>(double *__restrict xArray,
const int8_t *radem_array,
int dim1,
int start_row, int end_row);






template <typename T>
void singleVectorSORF(T cbuffer[], const int8_t *radem_array,
int repeat_position, int rademShape2,
int cbufferDim2) {
    T norm_constant = log2(cbufferDim2) / 2;
    norm_constant = 1 / pow(2, norm_constant);
    const int8_t *radem_element = radem_array + repeat_position;

    #pragma omp simd
    for (int i = 0; i < cbufferDim2; i++)
        cbuffer[i] *= radem_element[i] * norm_constant;

    radem_element += rademShape2;
    CPUHadamardTransformOps::singleVectorTransform<T>(cbuffer, cbufferDim2);

    #pragma omp simd
    for (int i = 0; i < cbufferDim2; i++)
        cbuffer[i] *= radem_element[i] * norm_constant;

    radem_element += rademShape2;
    CPUHadamardTransformOps::singleVectorTransform<T>(cbuffer, cbufferDim2);


    #pragma omp simd
    for (int i = 0; i < cbufferDim2; i++)
        cbuffer[i] *= radem_element[i] * norm_constant;

    CPUHadamardTransformOps::singleVectorTransform<T>(cbuffer, cbufferDim2);
}
//Explicitly instantiate for external use.
template void singleVectorSORF<double>(double cbuffer[], const int8_t *radem_array,
int repeat_position, int rademShape2,
int cbufferDim2);
template void singleVectorSORF<float>(float cbuffer[], const int8_t *radem_array,
int repeat_position, int rademShape2,
int cbufferDim2);





template <typename T>
void singleVectorRBFPostProcess(const T xdata[],
const T chi_arr[], double *output_array,
int dim2, int num_freqs,
int row_number, int repeat_num,
double scaling_term) {
    int output_start = repeat_num * dim2;
    T prodVal;
    double *__restrict xOut;
    const T *chiIn;
    int end_position = std::min(num_freqs, (repeat_num + 1) * dim2);
    end_position -= output_start;

    chiIn = chi_arr + output_start;
    xOut = output_array + 2 * output_start + row_number * 2 * num_freqs;

    for (int i=0; i < end_position; i++) {
        prodVal = xdata[i] * chiIn[i];
        *xOut += cos(prodVal) * scaling_term;
        xOut++;
        *xOut += sin(prodVal) * scaling_term;
        xOut++;
    }
}
template void singleVectorRBFPostProcess<double>(const double xdata[], const double chi_arr[],
double *output_array, int dim2, int num_freqs, int row_number, int repeat_num,
double scaling_term);
template void singleVectorRBFPostProcess<float>(const float xdata[], const float chi_arr[],
double *output_array, int dim2, int num_freqs, int row_number, int repeat_num,
double scaling_term);



template <typename T>
void singleVectorRBFPostGrad(const T xdata[],
const T chi_arr[], double *output_array,
double *gradient_array, double sigma,
int dim2, int num_freqs,
int row_number, int repeat_num,
double scaling_term) {
    int output_start = repeat_num * dim2;
    T prodVal, gradVal, cosVal, sinVal;
    double *__restrict xOut, *__restrict gradOut;
    const T *chiIn;
    int end_position = std::min(num_freqs, (repeat_num + 1) * dim2);
    end_position -= output_start;

    chiIn = chi_arr + output_start;
    xOut = output_array + 2 * output_start + row_number * 2 * num_freqs;
    gradOut = gradient_array + 2 * output_start + row_number * 2 * num_freqs;

    for (int i=0; i < end_position; i++) {
        gradVal = xdata[i] * chiIn[i];
        prodVal = gradVal * sigma;
        cosVal = cos(prodVal) * scaling_term;
        sinVal = sin(prodVal) * scaling_term;
        *xOut += cosVal;
        xOut++;
        *xOut += sinVal;
        xOut++;
        *gradOut -= sinVal * gradVal;
        gradOut++;
        *gradOut += cosVal * gradVal;
        gradOut++;
    }
}
// Explicitly instantiate for external use.
template void singleVectorRBFPostGrad<double>(
const double xdata[], const double chi_arr[],
double *output_array, double *gradient_array, double sigma,
int dim2, int num_freqs, int row_number, int repeat_num,
double scaling_term);

template void singleVectorRBFPostGrad<float>(
const float xdata[], const float chi_arr[],
double *output_array, double *gradient_array, double sigma,
int dim2, int num_freqs, int row_number, int repeat_num,
double scaling_term);

}  // namespace SharedCPURandomFeatureOps
