#ifndef FLOAT_RBF_OPS_H
#define FLOAT_RBF_OPS_H


struct ThreadRBFFloatArgs {
    int dim1, dim2;
    float *arrayStart;
    int startPosition, endPosition;
    int8_t *rademArray;
    double *outputArray;
    float *chiArr;
    int numFreqs;
    double rbfNormConstant;
};

struct ThreadRBFFloatGradArgs {
    int dim1, dim2;
    float *arrayStart;
    int startPosition, endPosition;
    int8_t *rademArray;
    double *outputArray;
    double *gradientArray;
    float *chiArr;
    int numFreqs;
    double rbfNormConstant;
    float sigma;
};


struct ThreadARDFloatGradArgs {
    int dim1;
    float *inputX;
    float *precompWeights;
    double *randomFeats;
    double *gradientArray;
    int32_t *sigmaMap;
    double *sigmaVals;
    int startPosition, endPosition;
    int numFreqs;
    int numLengthscales;
    double rbfNormConstant;
};

const char *rbfFeatureGenFloat_(float *cArray, int8_t *radem,
                float *chiArr, double *outputArray,
                double rbfNormConstant,
                int dim0, int dim1, int dim2,
                int numFreqs, int numThreads);

const char *rbfFloatGrad_(float *cArray, int8_t *radem,
                float *chiArr, double *outputArray,
                double *gradientArray,
                double rbfNormConstant, float sigma,
                int dim0, int dim1, int dim2,
                int numFreqs, int numThreads);

const char *ardFloatGrad_(float *inputX, double *randomFeatures,
        float *precompWeights, int32_t *sigmaMap, double *sigmaVals,
        double *gradient, int dim0, int dim1, int numLengthscales,
        int numFreqs, double rbfNormConstant, int numThreads);


void *ThreadRBFGenFloat(void *rowArgs);
void *ThreadRBFFloatGrad(void *rowArgs);
void *ThreadARDFloatGrad(void *rowArgs);


void rbfFloatFeatureGenLastStep_(float *xArray, float *chiArray,
        double *outputArray, double normConstant,
        int startRow, int endRow, int dim1,
        int dim2, int numFreqs);

void rbfFloatGradLastStep_(float *xArray, float *chiArray,
        double *outputArray, double *gradientArray,
        double normConstant, float sigma,
        int startRow, int endRow, int dim1,
        int dim2, int numFreqs);

void ardFloatGradCalcs_(float *inputX, double *randomFeatures,
        float *precompWeights, int32_t *sigmaMap, double *sigmaVals,
        double *gradient, int startRow, int endRow, int dim1,
        int numLengthscales, double rbfNormConstant, int numFreqs);

#endif
