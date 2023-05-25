#ifndef CPU_ARCCOS_SPECIFIC_CONVOLUTION_H
#define CPU_ARCCOS_SPECIFIC_CONVOLUTION_H

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

template <typename T>
const char *convArcCosFeatureGen_(int8_t *radem, T reshapedX[],
            T copyBuffer[], T chiArr[], double *outputArray,
            int numThreads, int reshapedDim0,
            int reshapedDim1, int reshapedDim2,
            int numFreqs, int rademShape2,
            int kernelOrder);

template <typename T>
void *threadConvArcCosGen(T reshapedXArray[], T copyBuffer[], T chiArr[],
        int8_t *rademArray, double *outputArray, int startRow,
        int endRow, int reshapedDim1, int reshapedDim2,
        int rademShape2, int numFreqs, int kernelOrder);

template <typename T>
void arcCosPostProcessOrder1(T reshapedX[], T chiArr[],
        double *outputArray, int reshapedDim1,
        int reshapedDim2, int numFreqs,
        int startRow, int endRow, int repeatNum);

template <typename T>
void arcCosPostProcessOrder2(T reshapedX[], T chiArr[],
        double *outputArray, int reshapedDim1,
        int reshapedDim2, int numFreqs,
        int startRow, int endRow, int repeatNum);

#endif
