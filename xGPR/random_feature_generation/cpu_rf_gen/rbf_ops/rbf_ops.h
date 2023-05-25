#ifndef SPEC_CPU_RBF_OPS_H
#define SPEC_CPU_RBF_OPS_H


template <typename T>
const char *rbfFeatureGen_(T cArray[], int8_t *radem,
                T chiArr[], double *outputArray,
                double rbfNormConstant,
                int dim0, int dim1, int dim2,
                int numFreqs, int numThreads);

template <typename T>
const char *rbfGrad_(T cArray[], int8_t *radem,
                T chiArr[], double *outputArray,
                double *gradientArray,
                double rbfNormConstant, T sigma,
                int dim0, int dim1, int dim2,
                int numFreqs, int numThreads);

template <typename T>
const char *ardGrad_(T inputX[], double *randomFeatures,
        T precompWeights[], int32_t *sigmaMap, double *sigmaVals,
        double *gradient, int dim0, int dim1, int numLengthscales,
        int numFreqs, double rbfNormConstant, int numThreads);


template <typename T>
void *ThreadRBFGen(T arrayStart[], int8_t *rademArray,
        T chiArr[], double *outputArray,
        int dim1, int dim2, int startPosition,
        int endPosition, int numFreqs, double rbfNormConstant);

template <typename T>
void *ThreadRBFGrad(T arrayStart[], int8_t* rademArray,
        T chiArr[], double *outputArray,
        double *gradientArray, int dim1, int dim2,
        int startPosition, int endPosition, int numFreqs,
        double rbfNormConstant, T sigma);

template <typename T>
void *ThreadARDGrad(T inputX[], double *randomFeats,
        T precompWeights[], int32_t *sigmaMap,
        double *sigmaVals, double *gradientArray,
        int startPosition, int endPosition,
        int dim1, int numLengthscales,
        int numFreqs, double rbfNormConstant);


template <typename T>
void rbfFeatureGenLastStep_(T xArray[], T chiArray[],
        double *outputArray, double normConstant,
        int startRow, int endRow, int dim1,
        int dim2, int numFreqs);

template <typename T>
void rbfGradLastStep_(T xArray[], T chiArray[],
        double *outputArray, double *gradientArray,
        double normConstant, T sigma,
        int startRow, int endRow, int dim1,
        int dim2, int numFreqs);

template <typename T>
void ardGradCalcs_(T inputX[], double *randomFeatures,
        T precompWeights[], int32_t *sigmaMap, double *sigmaVals,
        double *gradient, int startRow, int endRow, int dim1,
        int numLengthscales, double rbfNormConstant, int numFreqs);

#endif
