#ifndef RBF_CONVOLUTION_H
#define RBF_CONVOLUTION_H

#define MIN(a,b) ((a) < (b) ? (a) : (b))

struct ThreadConvRBFFloatArgs {
    int reshapedDim1, reshapedDim2;
    int numFreqs;
    float *reshapedXArray;
    float *chiArr;
    double *outputArray;
    float *copyBuffer;
    int8_t *rademArray;
    int startRow, endRow;
    double *gradientArray;
    float sigma;
    int rademShape2;
};


struct ThreadConvRBFDoubleArgs {
    int reshapedDim1, reshapedDim2;
    int numFreqs;
    double *reshapedXArray;
    double *chiArr;
    double *outputArray;
    double *copyBuffer;
    int8_t *rademArray;
    int startRow, endRow;
    double *gradientArray;
    double sigma;
    int rademShape2;
};


const char *doubleConvRBFFeatureGen_(int8_t *radem, double *reshapedX,
            double *copyBuffer, double *chiArr, double *outputArray,
            int numThreads, int reshapedDim0,
            int reshapedDim1, int reshapedDim2,
            int numFreqs, int rademShape2);

const char *floatConvRBFFeatureGen_(int8_t *radem, float *reshapedX,
            float *copyBuffer, float *chiArr, double *outputArray,
            int numThreads, int reshapedDim0,
            int reshapedDim1, int reshapedDim2,
            int numFreqs, int rademShape2);

const char *doubleConvRBFGrad_(int8_t *radem, double *reshapedX,
            double *copyBuffer, double *chiArr, double *outputArray,
            double *gradientArray, double sigma,
            int numThreads, int reshapedDim0,
            int reshapedDim1, int reshapedDim2,
            int numFreqs, int rademShape2);

const char *floatConvRBFGrad_(int8_t *radem, float *reshapedX,
            float *copyBuffer, float *chiArr, double *outputArray,
            double *gradientArray, float sigma,
            int numThreads, int reshapedDim0,
            int reshapedDim1, int reshapedDim2,
            int numFreqs, int rademShape2);

void *floatThreadConvRBFGen(void *sharedArgs);

void *doubleThreadConvRBFGen(void *sharedArgs);

void *floatThreadConvRBFGrad(void *sharedArgs);

void *doubleThreadConvRBFGrad(void *sharedArgs);

void doubleRBFPostProcess(double *reshapedX, double *chiArr,
        double *outputArray, int reshapedDim1,
        int reshapedDim2, int numFreqs,
        int startRow, int endRow, int repeatNum);

void floatRBFPostProcess(float *reshapedX, float *chiArr,
        double *outputArray, int reshapedDim1,
        int reshapedDim2, int numFreqs,
        int startRow, int endRow, int repeatNum);

void doubleRBFPostGrad(double *reshapedX, double *chiArr,
        double *outputArray, double *gradientArray,
        int reshapedDim1, int reshapedDim2,
        int numFreqs, int startRow, int endRow,
        int repeatNum, double sigma);

void floatRBFPostGrad(float *reshapedX, float *chiArr,
        double *outputArray, double *gradientArray,
        int reshapedDim1, int reshapedDim2,
        int numFreqs, int startRow, int endRow,
        int repeatNum, float sigma);
#endif
