#ifndef CUDA_RBF_SPECIFIC_CONVOLUTION_H
#define CUDA_RBF_SPECIFIC_CONVOLUTION_H

#include <stdint.h>

#define MIN(a,b) ((a) < (b) ? (a) : (b))

template <typename T>
const char *convRBFFeatureGen(const int8_t *radem, const T xdata[],
            const T chiArr[], double *outputArray,
            int dim0, int dim1, int dim2, int numFreqs,
            int rademShape2, int convWidth, int paddedBufferSize,
            double scalingTerm);

template <typename T>
const char *convRBFFeatureGrad(const int8_t *radem, const T xdata[],
            const T chiArr[], double *outputArray,
            double *gradientArray, double sigma,
            int dim0, int dim1, int dim2, int numFreqs,
            int rademShape2, int convWidth, int paddedBufferSize,
            double scalingTerm);

#endif
