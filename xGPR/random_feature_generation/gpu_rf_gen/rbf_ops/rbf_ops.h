#ifndef CUDA_RBF_SPEC_OPERATIONS_H
#define CUDA_RBF_SPEC_OPERATIONS_H

template <typename T>
const char *RBFFeatureGen(T cArray[], int8_t *radem,
                T chiArr[], double *outputArray,
                double rbfNormConstant,
                int dim0, int dim1, int dim2,
                int numFreqs);

template <typename T>
const char *RBFFeatureGrad(T cArray[], int8_t *radem,
                T chiArr[], double *outputArray,
                double *gradientArray, double rbfNormConstant,
                T sigma, int dim0, int dim1, int dim2,
                int numFreqs);

template <typename T>
const char *ardCudaGrad(T inputX[], double *randomFeats,
                T precompWeights[], int32_t *sigmaMap,
                double *sigmaVals, double *gradient, int dim0,
                int dim1, int numLengthscales, int numFreqs,
                double rbfNormConstant);

#endif
