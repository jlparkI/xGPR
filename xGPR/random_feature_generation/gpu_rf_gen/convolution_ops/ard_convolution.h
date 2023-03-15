#ifndef CUDA_ARD_SPECIFIC_CONVOLUTION_H
#define CUDA_ARD_SPECIFIC_CONVOLUTION_H

const char *ardConvCudaDoubleGrad(double *inputX, double *randomFeats,
                double *precompWeights, int32_t *sigmaMap,
                double *sigmaVals, double *gradient, int dim0,
                int dim1, int dim2, int numLengthscales,
                int numFreqs, double rbfNormConstant);

const char *ardConvCudaFloatGrad(float *inputX, double *randomFeats,
                float *precompWeights, int32_t *sigmaMap,
                double *sigmaVals, double *gradient, int dim0,
                int dim1, int dim2, int numLengthscales,
                int numFreqs, double rbfNormConstant);


#endif
