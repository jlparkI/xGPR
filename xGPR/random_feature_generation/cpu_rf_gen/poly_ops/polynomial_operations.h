#ifndef CPU_POLYNOMIAL_OPERATIONS_H
#define CPU_POLYNOMIAL_OPERATIONS_H


template <typename T>
const char *cpuExactQuadratic_(T inArray[], double *outArray,
                int inDim0, int inDim1, int numThreads);

template <typename T>
void *ThreadExactQuadratic(T inArray[], double *outArray, int startPosition,
        int endPosition, int inDim0, int inDim1);

template <typename T>
void exactQuadraticFeatureGen(T inArray[], double *outArray, int startRow,
                    int endRow, int inDim0, int inDim1);

#endif
