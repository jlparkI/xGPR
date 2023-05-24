#ifndef CUDA_RBF_SPECIFIC_CONVOLUTION_H
#define CUDA_RBF_SPECIFIC_CONVOLUTION_H

#define MIN(a,b) ((a) < (b) ? (a) : (b))

template <typename T>
const char *convRBFFeatureGen(int8_t *radem, T reshapedX[],
            T featureArray[], T chiArr[], double *outputArray,     
            int reshapedDim0, int reshapedDim1, int reshapedDim2,
            int numFreqs, int rademShape2, double scalingTerm);

template <typename T>
const char *convRBFFeatureGrad(int8_t *radem, T reshapedX[],
            T featureArray[], T chiArr[], double *outputArray,     
            double *gradientArray, double sigma,
            int reshapedDim0, int reshapedDim1, int reshapedDim2,
            int numFreqs, int rademShape2, double scalingTerm);

#endif
