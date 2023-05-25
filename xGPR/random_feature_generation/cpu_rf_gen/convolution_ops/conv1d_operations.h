#ifndef CONV1D_OPERATIONS_H
#define CONV1D_OPERATIONS_H



template <typename T>
const char *conv1dPrep_(int8_t *radem, T reshapedX[],
            int numThreads, int reshapedDim0,
            int reshapedDim1, int reshapedDim2,
            int startPosition, int numFreqs);

template <typename T>
void *threadConv1d(T reshapedXArray[], int8_t* rademArray,
        int reshapedDim1, int reshapedDim2, int numFreqs,
        int startRow, int endRow, int startPosition);

#endif
