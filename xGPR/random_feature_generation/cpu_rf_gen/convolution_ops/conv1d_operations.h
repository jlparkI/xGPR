#ifndef CONV1D_OPERATIONS_H
#define CONV1D_OPERATIONS_H


#include <stdint.h>

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))


template <typename T>
const char *conv1dMaxpoolFeatureGen_(int8_t *radem, T xdata[],
            T chiArr[], double *outputArray, int32_t *seqlengths,
            int xdim0, int xdim1, int xdim2, int numThreads, int numFreqs,
            int convWidth, int paddedBufferSize,
            bool simplex);

template <typename T>
void *allInOneConvMaxpoolGen(T xdata[], int8_t *rademArray, T chiArr[],
        double *outputArray, int32_t *seqlengths, int dim1, int dim2,
        int numFreqs, int startRow, int endRow,
        int convWidth, int paddedBufferSize);

template <typename T>
void *allInOneConvMaxpoolSimplex(T xdata[], int8_t *rademArray, T chiArr[],
        double *outputArray, int32_t *seqlengths, int dim1, int dim2,
        int numFreqs, int startRow, int endRow,
        int convWidth, int paddedBufferSize);

template <typename T>
void singleVectorMaxpoolPostProcess(const T xdata[],
        const T chiArr[], double *outputArray,
        int dim2, int numFreqs,
        int rowNumber, int repeatNum);

#endif
