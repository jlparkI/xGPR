#ifndef CUDA_ARCCOS_SPECIFIC_CONVOLUTION_H
#define CUDA_ARCCOS_SPECIFIC_CONVOLUTION_H

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))


const char *floatConvArcCosFeatureGen(int8_t *radem, float *reshapedX,
            float *featureArray, float *chiArr, double *outputArray,     
            int reshapedDim0, int reshapedDim1, int reshapedDim2,
            int numFreqs, int rademShape2, double scalingTerm,
            int kernelOrder);

const char *doubleConvArcCosFeatureGen(int8_t *radem, double *reshapedX,
                double *featureArray, double *chiArr, double *outputArray,
                int reshapedDim0, int reshapedDim1, int reshapedDim2,
                int numFreqs, int rademShape2, double scalingTerm,
                int kernelOrder);

#endif
