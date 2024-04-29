#ifndef CUDA_ARD_SPEC_OPERATIONS_H
#define CUDA_ARD_SPEC_OPERATIONS_H

#include <stdint.h>

template <typename T>
const char *ardCudaGrad(T inputX[], double *randomFeats,
                T precompWeights[], int32_t *sigmaMap,
                double *sigmaVals, double *gradient, int dim0,
                int dim1, int numLengthscales, int numFreqs,
                double rbfNormConstant);

#endif
