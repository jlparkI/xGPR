#ifndef FLOAT_CUDA_RBF_SPEC_OPERATIONS_H
#define FLOAT_CUDA_RBF_SPEC_OPERATIONS_H

const char *floatRBFFeatureGen(float *cArray, int8_t *radem,
                float *chiArr, double *outputArray,
                double rbfNormConstant,
                int dim0, int dim1, int dim2,
                int numFreqs);

const char *floatRBFFeatureGrad(float *cArray, int8_t *radem,
                float *chiArr, double *outputArray,
                double *gradientArray, double rbfNormConstant,
                float sigma, int dim0, int dim1, int dim2,
                int numFreqs);

#endif
