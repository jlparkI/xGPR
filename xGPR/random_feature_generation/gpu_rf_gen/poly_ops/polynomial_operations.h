#ifndef CUDA_POLYNOMIAL_OPERATIONS_H
#define CUDA_POLYNOMIAL_OPERATIONS_H
#include <stdint.h>

template <typename T>
const char *cudaExactQuadratic_(T inArray[], double *outArray, 
                    int inDim0, int inDim1);

template <typename T>
const char *approxPolynomial_(int8_t *radem, T reshapedX[],
        T copyBuffer[], T chiArr[], double *outArray,
        int polydegree, int reshapedDim0, int reshapedDim1,
        int reshapedDim2, int numFreqs);

#endif
