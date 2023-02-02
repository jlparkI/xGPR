#ifndef THREAD_ARGS_H
#define THREAD_ARGS_H

struct ThreadSORFFloatArrayArgs {
    int dim1, dim2;
    float *arrayStart;
    int startPosition, endPosition;
    int8_t *rademArray;
};


struct Thread3DFloatArrayArgs {
    int dim1, dim2;
    float *arrayStart;
    int startPosition, endPosition;
};

struct ThreadSORFDoubleArrayArgs {
    int dim1, dim2;
    double *arrayStart;
    int startPosition, endPosition;
    int8_t *rademArray;
};


struct Thread3DDoubleArrayArgs {
    int dim1, dim2;
    double *arrayStart;
    int startPosition, endPosition;
};


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
#endif
