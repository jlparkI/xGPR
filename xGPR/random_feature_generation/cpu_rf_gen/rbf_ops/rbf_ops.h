#ifndef SPEC_CPU_RBF_OPS_H
#define SPEC_CPU_RBF_OPS_H

#include <stdint.h>


template <typename T>
const char *rbfFeatureGen_(T cArray[], int8_t *radem,
                T chiArr[], double *outputArray,
                double rbfNormConstant,
                int dim0, int dim1, int rademShape2,
                int numFreqs, int numThreads,
                int paddedBufferSize);

template <typename T>
const char *rbfGrad_(T cArray[], int8_t *radem,
                T chiArr[], double *outputArray,
                double *gradientArray,
                double rbfNormConstant, T sigma,
                int dim0, int dim1, int rademShape2,
                int numFreqs, int numThreads,
                int paddedBufferSize);


template <typename T>
void *allInOneRBFGen(T xdata[], int8_t *rademArray, T chiArr[],
        double *outputArray, int dim1, int numFreqs, int rademShape2,
        int startRow, int endRow, int paddedBufferSize,
        double scalingTerm);


template <typename T>
void *allInOneRBFGrad(T xdata[], int8_t *rademArray, T chiArr[],
        double *outputArray, double *gradientArray,
        int dim1, int numFreqs, int rademShape2, int startRow,
        int endRow, int paddedBufferSize,
        double scalingTerm, T sigma);


template <typename T>
const char *ardGrad_(T inputX[], double *randomFeatures,
        T precompWeights[], int32_t *sigmaMap, double *sigmaVals,
        double *gradient, int dim0, int dim1, int numLengthscales,
        int numFreqs, double rbfNormConstant, int numThreads);



template <typename T>
void *ThreadARDGrad(T inputX[], double *randomFeatures,
        T precompWeights[], int32_t *sigmaMap,
        double *sigmaVals, double *gradient,
        int startRow, int endRow,
        int dim1, int numLengthscales,
        int numFreqs, double rbfNormConstant);

#endif
