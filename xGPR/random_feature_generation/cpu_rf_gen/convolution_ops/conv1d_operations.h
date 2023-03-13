#ifndef CONV1D_OPERATIONS_H
#define CONV1D_OPERATIONS_H

struct ThreadConv1dDoubleArgs {
    int startPosition;
    int reshapedDim1, reshapedDim2;
    int numFreqs;
    double *reshapedXArray;
    int8_t *rademArray;
    int startRow, endRow;
};


struct ThreadConv1dFloatArgs {
    int startPosition;
    int reshapedDim1, reshapedDim2;
    int numFreqs;
    float *reshapedXArray;
    int8_t *rademArray;
    int startRow, endRow;
};


const char *doubleConv1dPrep_(int8_t *radem, double *reshapedX,
            int numThreads, int reshapedDim0,
            int reshapedDim1, int reshapedDim2,
            int startPosition, int numFreqs);

const char *floatConv1dPrep_(int8_t *radem, float *reshapedX,
            int numThreads, int reshapedDim0,
            int reshapedDim1, int reshapedDim2,
            int startPosition, int numFreqs);

void *doubleThreadConv1d(void *rowArgs);

void *floatThreadConv1d(void *rowArgs);

#endif
