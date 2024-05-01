#ifndef SPEC_CPU_ARD_OPS_H
#define SPEC_CPU_ARD_OPS_H

#include <stdint.h>


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
