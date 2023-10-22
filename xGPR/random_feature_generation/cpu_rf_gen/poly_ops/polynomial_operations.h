#ifndef CPU_POLYNOMIAL_OPERATIONS_H
#define CPU_POLYNOMIAL_OPERATIONS_H
#include <stdint.h>


template <typename T>
const char *cpuExactQuadratic_(T inArray[], double *outArray,
                int inDim0, int inDim1, int numThreads);

template <typename T>
void *ThreadExactQuadratic(T inArray[], double *outArray, int startPosition,
        int endPosition, int inDim0, int inDim1);

template <typename T>
const char *approxPolynomial_(int8_t *radem, T reshapedX[],
            T copyBuffer[], T chiArr[], double *outputArray,
            int numThreads, int polydegree, int reshapedDim0,
            int reshapedDim1, int reshapedDim2,
            int numFreqs);

template <typename T>
void *threadApproxPolynomial(T inArray[], T copyBuffer[], int8_t *radem,
        T chiArr[], double *outputArray, int polydegree, int dim1,
        int dim2, int numFreqs, int startRow, int endRow);

template <typename T>
void *outArrayMatTransfer(T copyBuffer[], double *outArray, T chiArr[],
        int dim1, int dim2, int numFreqs, int startRow, int endRow,
        int chiArrRow);

template <typename T>
void *outArrayCopyTransfer(T copyBuffer[], double *outArray, T chiArr[],
        int dim1, int dim2, int numFreqs, int startRow, int endRow);

#endif
