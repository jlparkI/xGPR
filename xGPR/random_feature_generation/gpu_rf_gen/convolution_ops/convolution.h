#ifndef CUDA_CONVOLUTION_H
#define CUDA_CONVOLUTION_H

#include <stdint.h>


template <typename T>
const char *conv1dPrep(int8_t *radem, T reshapedX[], int reshapedDim0, 
                int reshapedDim1, int reshapedDim2, int startPosition,
                int numFreqs);


template <typename T>
const char *conv1dMaxpoolFeatureGen(const int8_t *radem, const T xdata[],
            const T chiArr[], double *outputArray, const int32_t *seqlengths,
            int xdim0, int xdim1, int xdim2, int numFreqs,
            int convWidth, int paddedBufferSize);

#endif
