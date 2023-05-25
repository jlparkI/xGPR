#ifndef CUDA_ARCCOS_SPECIFIC_CONVOLUTION_H
#define CUDA_ARCCOS_SPECIFIC_CONVOLUTION_H

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))


template <typename T>
const char *convArcCosFeatureGen(int8_t *radem, T reshapedX[],
            T featureArray[], T chiArr[], double *outputArray,     
            int reshapedDim0, int reshapedDim1, int reshapedDim2,
            int numFreqs, int rademShape2, double scalingTerm,
            int kernelOrder);

#endif
