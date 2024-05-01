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
            double scalingTerm, int scalingType,
            bool simplex);

template <typename T>
const char *convRBFGrad_(int8_t *radem, T xdata[],
            T chiArr[], double *outputArray,
            int32_t *seqlengths,
            double *gradientArray, T sigma,
            int numThreads, int dim0,
            int dim1, int dim2, int numFreqs,
            int rademShape2, int convWidth,
            int paddedBufferSize,
            double scalingTerm, int scalingType,
            bool simplex);

template <typename T>
void *allInOneConvRBFGen(T xdata[], int8_t *rademArray, T chiArr[],
        double *outputArray, int32_t *seqlengths, int dim1, int dim2,
        int numFreqs, int rademShape2, int startRow, int endRow,
        int convWidth, int paddedBufferSize,
        double scalingTerm, int scalingType);

template <typename T>
void *allInOneConvRBFSimplex(T xdata[], int8_t *rademArray, T chiArr[],
        double *outputArray, int32_t *seqlengths, int dim1, int dim2,
        int numFreqs, int rademShape2, int startRow, int endRow,
        int convWidth, int paddedBufferSize,
        double scalingTerm, int scalingType);

template <typename T>
void *allInOneConvRBFGrad(T xdata[], int8_t *rademArray, T chiArr[],
        double *outputArray, int32_t *seqlengths, double *gradientArray,
        int dim1, int dim2, int numFreqs, int rademShape2, int startRow,
        int endRow, int convWidth, int paddedBufferSize,
        double scalingTerm, int scalingType, T sigma);

template <typename T>
void *allInOneConvRBFGradSimplex(T xdata[], int8_t *rademArray, T chiArr[],
        double *outputArray, int32_t *seqlengths, double *gradientArray,
        int dim1, int dim2, int numFreqs, int rademShape2, int startRow,
        int endRow, int convWidth, int paddedBufferSize,
        double scalingTerm, int scalingType, T sigma);

#endif
