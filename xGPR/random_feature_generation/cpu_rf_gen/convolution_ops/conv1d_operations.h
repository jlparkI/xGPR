#ifndef CONV1D_OPERATIONS_H
#define CONV1D_OPERATIONS_H

#define VALID_INPUTS 0
#define INVALID_INPUTS 1
#define EPS_TOLERANCE 0.0
#define MAX_THREADS 8

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



struct ThreadConvRBFFloatArgs {
    int startPosition;
    int reshapedDim1, reshapedDim2;
    int numFreqs;
    float *reshapedXArray;
    float *chiArr;
    double *outputArray;
    float *copyBuffer;
    int8_t *rademArray;
    int startRow, endRow;
    double *gradientArray;
};


struct ThreadConvRBFDoubleArgs {
    int startPosition;
    int reshapedDim1, reshapedDim2;
    int numFreqs;
    double *reshapedXArray;
    double *chiArr;
    double *outputArray;
    double *copyBuffer;
    int8_t *rademArray;
    int startRow, endRow;
    double *gradientArray;
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


const char *doubleConvRBFFeatureGen_(int8_t *radem, double *reshapedX,
            double *copyBuffer, double *chiArr, double *outputArray,
            int numThreads, int reshapedDim0,
            int reshapedDim1, int reshapedDim2,
            int startPosition, int numFreqs);

const char *floatConvRBFFeatureGen_(int8_t *radem, float *reshapedX,
            float *copyBuffer, float *chiArr, double *outputArray,
            int numThreads, int reshapedDim0,
            int reshapedDim1, int reshapedDim2,
            int startPosition, int numFreqs);

const char *doubleConvRBFGrad_(int8_t *radem, double *reshapedX,
            double *copyBuffer, double *chiArr, double *outputArray,
            double *gradientArray,
            int numThreads, int reshapedDim0,
            int reshapedDim1, int reshapedDim2,
            int startPosition, int numFreqs);

const char *floatConvRBFGrad_(int8_t *radem, float *reshapedX,
            float *copyBuffer, float *chiArr, double *outputArray,
            double *gradientArray,
            int numThreads, int reshapedDim0,
            int reshapedDim1, int reshapedDim2,
            int startPosition, int numFreqs);

void *floatThreadConvRBFGen(void *sharedArgs);

void *doubleThreadConvRBFGen(void *sharedArgs);

void *floatThreadConvRBFGrad(void *sharedArgs);

void *doubleThreadConvRBFGrad(void *sharedArgs);

void doubleRBFGenPostProcess(double *reshapedX, double *chiArr,
        double *outputArray, int reshapedDim0, int reshapedDim1,
        int reshapedDim2, int startPosition, int numFreqs,
        int startRow, int endRow);

void floatRBFGenPostProcess(float *reshapedX, float *chiArr,
        double *outputArray, int reshapedDim0, int reshapedDim1,
        int reshapedDim2, int startPosition, int numFreqs,
        int startRow, int endRow);

void doubleRBFGenGrad(double *reshapedX, double *chiArr,
        double *outputArray, double *gradientArray,
        int reshapedDim0, int reshapedDim1,
        int reshapedDim2, int startPosition, int numFreqs,
        int startRow, int endRow);

void floatRBFGenGrad(float *reshapedX, float *chiArr,
        double *outputArray, double *gradientArray,
        int reshapedDim0, int reshapedDim1,
        int reshapedDim2, int startPosition, int numFreqs,
        int startRow, int endRow);
#endif
