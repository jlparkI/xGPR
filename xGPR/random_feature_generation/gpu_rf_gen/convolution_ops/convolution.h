#ifndef CUDA_CONVOLUTION_H
#define CUDA_CONVOLUTION_H

const char *floatConv1dPrep(int8_t *radem, float *reshapedX, int reshapedDim0, 
                int reshapedDim1, int reshapedDim2, int startPosition,
                int numFreqs);

const char *doubleConv1dPrep(int8_t *radem, double *reshapedX, int reshapedDim0, 
                int reshapedDim1, int reshapedDim2, int startPosition,
                int numFreqs);

const char *floatConvRBFFeatureGen(int8_t *radem, float *reshapedX,
            float *sinFeatures, float *cosFeatures, float *chiArr,        
            int reshapedDim0, int reshapedDim1, int reshapedDim2,
            int startPosition, int numFreqs);

const char *doubleConvRBFFeatureGen(int8_t *radem, double *reshapedX,
                double *sinFeatures, double *cosFeatures,
                double *chiArr, int reshapedDim0, 
                int reshapedDim1, int reshapedDim2, int startPosition,
                int numFreqs);
#endif
