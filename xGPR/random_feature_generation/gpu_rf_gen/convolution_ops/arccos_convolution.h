#ifndef CUDA_RBF_SPECIFIC_CONVOLUTION_H
#define CUDA_RBF_SPECIFIC_CONVOLUTION_H

#define MIN(a,b) ((a) < (b) ? (a) : (b))


const char *floatConvRBFFeatureGen(int8_t *radem, float *reshapedX,
            float *featureArray, float *chiArr, double *outputArray,     
            int reshapedDim0, int reshapedDim1, int reshapedDim2,
            int numFreqs, int rademShape2, double scalingTerm);

const char *doubleConvRBFFeatureGen(int8_t *radem, double *reshapedX,
                double *featureArray, double *chiArr, double *outputArray,
                int reshapedDim0, int reshapedDim1, int reshapedDim2,
                int numFreqs, int rademShape2, double scalingTerm);

const char *floatConvRBFFeatureGrad(int8_t *radem, float *reshapedX,
            float *featureArray, float *chiArr, double *outputArray,     
            double *gradientArray, double sigma,
            int reshapedDim0, int reshapedDim1, int reshapedDim2,
            int numFreqs, int rademShape2, double scalingTerm);

const char *doubleConvRBFFeatureGrad(int8_t *radem, double *reshapedX,
                double *featureArray, double *chiArr, double *outputArray,
                double *gradientArray, double sigma,
                int reshapedDim0, int reshapedDim1, int reshapedDim2,
                int numFreqs, int rademShape2, double scalingTerm);

#endif
