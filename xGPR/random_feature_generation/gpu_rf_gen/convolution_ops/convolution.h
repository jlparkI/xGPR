#ifndef CUDA_CONVOLUTION_H
#define CUDA_CONVOLUTION_H

#include <stdint.h>


template <typename T>
const char *conv1dMaxpoolFeatureGen(const int8_t *radem, const T xdata[],
            const T chiArr[], double *outputArray, const int32_t *seqlengths,
            int xdim0, int xdim1, int xdim2, int numFreqs,
            int convWidth, int paddedBufferSize, int rademShape2);

#endif
