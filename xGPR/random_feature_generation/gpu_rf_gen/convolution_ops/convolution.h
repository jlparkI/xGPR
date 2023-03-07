#ifndef CUDA_CONVOLUTION_H
#define CUDA_CONVOLUTION_H

const char *floatConv1dPrep(int8_t *radem, float *reshapedX, int reshapedDim0, 
                int reshapedDim1, int reshapedDim2, int startPosition,
                int numFreqs);

const char *doubleConv1dPrep(int8_t *radem, double *reshapedX, int reshapedDim0, 
                int reshapedDim1, int reshapedDim2, int startPosition,
                int numFreqs);

const char *floatConvRBFFeatureGen(int8_t *radem, float *reshapedX,
            float *featureArray, float *chiArr, double *outputArray,     
            int reshapedDim0, int reshapedDim1, int reshapedDim2,
            int numFreqs, double scalingTerm);

const char *doubleConvRBFFeatureGen(int8_t *radem, double *reshapedX,
                double *featureArray, double *chiArr, double *outputArray,
                int reshapedDim0, int reshapedDim1, int reshapedDim2,
                int numFreqs, double scalingTerm);

void rbfConvFloatPostProcess(float *featureArray, float *chiArr,
        double *outputArray, int reshapedDim0, int reshapedDim1,
        int reshapedDim2, int startPosition, int numFreqs,
        double scalingTerm);

void rbfConvDoublePostProcess(double *featureArray, double *chiArr,
        double *outputArray, int reshapedDim0, int reshapedDim1,
        int reshapedDim2, int startPosition, int numFreqs,
        double scalingTerm);
#endif
