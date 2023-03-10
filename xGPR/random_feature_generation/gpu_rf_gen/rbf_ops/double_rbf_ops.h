#ifndef DOUBLE_CUDA_RBF_SPEC_OPERATIONS_H
#define DOUBLE_CUDA_RBF_SPEC_OPERATIONS_H

const char *doubleRBFFeatureGen(double *cArray, int8_t *radem,
                double *chiArr, double *outputArray,
                double rbfNormConstant,
                int dim0, int dim1, int dim2,
                int numFreqs);

const char *doubleRBFFeatureGrad(double *cArray, int8_t *radem,
                double *chiArr, double *outputArray,
                double *gradientArray, double rbfNormConstant,
                double sigma, int dim0, int dim1, int dim2,
                int numFreqs);

const char *ardCudaDoubleGrad(double *inputX, double *randomFeats,
                double *precompWeights, int32_t *sigmaMap,
                double *sigmaVals, double *gradient, int dim0,
                int dim1, int numLengthscales, int numFreqs,
                double rbfNormConstant);

#endif
