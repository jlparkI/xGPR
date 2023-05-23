#ifndef ARCCOS_CONVOLUTION_H
#define ARCCOS_CONVOLUTION_H

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

struct ThreadConvArcCosFloatArgs {
    int reshapedDim1, reshapedDim2;
    int numFreqs;
    float *reshapedXArray;
    float *chiArr;
    double *outputArray;
    float *copyBuffer;
    int8_t *rademArray;
    int startRow, endRow;
    double *gradientArray;
    int rademShape2;
    int kernelOrder;
};


struct ThreadConvArcCosDoubleArgs {
    int reshapedDim1, reshapedDim2;
    int numFreqs;
    double *reshapedXArray;
    double *chiArr;
    double *outputArray;
    double *copyBuffer;
    int8_t *rademArray;
    int startRow, endRow;
    double *gradientArray;
    int rademShape2;
    int kernelOrder;
};


const char *doubleConvArcCosFeatureGen_(int8_t *radem, double *reshapedX,
            double *copyBuffer, double *chiArr, double *outputArray,
            int numThreads, int reshapedDim0,
            int reshapedDim1, int reshapedDim2,
            int numFreqs, int rademShape2,
            int kernelOrder);

const char *floatConvArcCosFeatureGen_(int8_t *radem, float *reshapedX,
            float *copyBuffer, float *chiArr, double *outputArray,
            int numThreads, int reshapedDim0,
            int reshapedDim1, int reshapedDim2,
            int numFreqs, int rademShape2,
            int kernelOrder);


void *floatThreadConvArcCosGen(void *sharedArgs);

void *doubleThreadConvArcCosGen(void *sharedArgs);

void doubleArcCosPostProcessOrder1(double *reshapedX, double *chiArr,
        double *outputArray, int reshapedDim1,
        int reshapedDim2, int numFreqs,
        int startRow, int endRow, int repeatNum);

void floatArcCosPostProcessOrder1(float *reshapedX, float *chiArr,
        double *outputArray, int reshapedDim1,
        int reshapedDim2, int numFreqs,
        int startRow, int endRow, int repeatNum);

void doubleArcCosPostProcessOrder2(double *reshapedX, double *chiArr,
        double *outputArray, int reshapedDim1,
        int reshapedDim2, int numFreqs,
        int startRow, int endRow, int repeatNum);

void floatArcCosPostProcessOrder2(float *reshapedX, float *chiArr,
        double *outputArray, int reshapedDim1,
        int reshapedDim2, int numFreqs,
        int startRow, int endRow, int repeatNum);
#endif
