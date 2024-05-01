#ifndef CUDA_RBF_SPEC_OPERATIONS_H
#define CUDA_RBF_SPEC_OPERATIONS_H

#include <stdint.h>

template <typename T>
const char *RBFFeatureGen(T cArray[], int8_t *radem,
                T chiArr[], double *outputArray,
                double rbfNormConstant,
                int dim0, int dim1, int rademShape2,
                int numFreqs, int paddedBufferSize,
                bool simplex);

template <typename T>
const char *RBFFeatureGrad(T cArray[], int8_t *radem,
                T chiArr[], double *outputArray,
                double *gradientArray, double rbfNormConstant,
                T sigma, int dim0, int dim1, int rademShape2,
                int numFreqs, int paddedBufferSize,
                bool simplex);

#endif
