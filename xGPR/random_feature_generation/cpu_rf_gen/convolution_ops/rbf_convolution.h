#ifndef RBF_CONVOLUTION_H
#define RBF_CONVOLUTION_H

#define MIN(a,b) ((a) < (b) ? (a) : (b))


template <typename T>
const char *convRBFFeatureGen_(int8_t *radem, T reshapedX[],
            T copyBuffer[], T chiArr[], double *outputArray,
            int numThreads, int reshapedDim0,
            int reshapedDim1, int reshapedDim2,
            int numFreqs, int rademShape2);

template <typename T>
const char *convRBFGrad_(int8_t *radem, T reshapedX[],
            T copyBuffer[], T chiArr[], double *outputArray,
            double *gradientArray, T sigma,
            int numThreads, int reshapedDim0,
            int reshapedDim1, int reshapedDim2,
            int numFreqs, int rademShape2);

template <typename T>
void *threadConvRBFGen(T reshapedXArray[], T copyBuffer[],
        int8_t *rademArray, T chiArr[], double *outputArray,
        int reshapedDim1, int reshapedDim2, int numFreqs,
        int rademShape2, int startRow, int endRow);

template <typename T>
void *threadConvRBFGrad(T reshapedXArray[], T copyBuffer[],
        int8_t *rademArray, T chiArr[], double *outputArray,
        double *gradientArray, int reshapedDim1,
        int reshapedDim2, int numFreqs, int rademShape2,
        int startRow, int endRow, T sigma);

template <typename T>
void RBFPostProcess(T reshapedX[], T chiArr[],
        double *outputArray, int reshapedDim1,
        int reshapedDim2, int numFreqs,
        int startRow, int endRow, int repeatNum);


template <typename T>
void RBFPostGrad(T reshapedX[], T chiArr[],
        double *outputArray, double *gradientArray,
        int reshapedDim1, int reshapedDim2,
        int numFreqs, int startRow, int endRow,
        int repeatNum, T sigma);

#endif
