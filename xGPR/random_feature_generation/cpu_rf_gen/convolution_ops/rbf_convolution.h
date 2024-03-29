#ifndef RBF_CONVOLUTION_H
#define RBF_CONVOLUTION_H

#include <stdint.h>

#define MIN(a,b) ((a) < (b) ? (a) : (b))


template <typename T>
const char *convRBFFeatureGen_(int8_t *radem, T xdata[],
            T chiArr[], double *outputArray,
            int32_t *seqlengths,
            int numThreads, int dim0,
            int dim1, int dim2,
            int numFreqs, int rademShape2,
            int convWidth, int paddedBufferSize,
            double scalingTerm, int scalingType);

template <typename T>
const char *convRBFGrad_(int8_t *radem, T xdata[],
            T chiArr[], double *outputArray,
            int32_t *seqlengths,
            double *gradientArray, T sigma,
            int numThreads, int dim0,
            int dim1, int dim2, int numFreqs,
            int rademShape2, int convWidth,
            int paddedBufferSize,
            double scalingTerm, int scalingType);

template <typename T>
void *threadConvRBFGen(T xdata[], T copyBuffer[],
        int8_t *rademArray, T chiArr[], double *outputArray,
        int32_t *seqlengths, int dim1, int dim2, int numFreqs,
        int rademShape2, int startRow, int endRow,
        int convWidth, int paddedBufferSize,
        double scalingTerm, int scalingType);

template <typename T>
void *threadConvRBFGrad(T xdata[], T copyBuffer[],
        int8_t *rademArray, T chiArr[], double *outputArray,
        int32_t *seqlengths, double *gradientArray, int dim1,
        int dim2, int numFreqs, int rademShape2,
        int startRow, int endRow, T sigma,
        int convWidth, int paddedBufferSize,
        double scalingTerm, int scalingType);

template <typename T>
void RBFPostProcess(const T __restrict xdata[],
        const T chiArr[], double *__restrict outputArray,
        int dim1, int dim2, int numFreqs,
        int startRow, int endRow, int repeatNum,
        int convWidth, const int32_t *seqlengths,
        double scalingTerm, int scalingType);


template <typename T>
void RBFPostGrad(const T __restrict xdata[],
        const T chiArr[], double *__restrict outputArray,
        double *__restrict gradientArray,
        int dim1, int dim2,
        int numFreqs, int startRow, int endRow,
        int repeatNum, T sigma, int convWidth,
        const int32_t *seqlengths,
        double scalingTerm, int scalingType);

#endif
