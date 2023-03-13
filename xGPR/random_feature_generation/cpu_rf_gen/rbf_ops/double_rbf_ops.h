#ifndef DOUBLE_RBF_OPS_H
#define DOUBLE_RBF_OPS_H


struct ThreadRBFDoubleArgs {
    int dim1, dim2;
    double *arrayStart;
    int startPosition, endPosition;
    int8_t *rademArray;
    double *outputArray;
    double *chiArr;
    int numFreqs;
    double rbfNormConstant;
};

struct ThreadRBFDoubleGradArgs {
    int dim1, dim2;
    double *arrayStart;
    int startPosition, endPosition;
    int8_t *rademArray;
    double *outputArray;
    double *gradientArray;
    double *chiArr;
    int numFreqs;
    double rbfNormConstant;
    double sigma;
};


struct ThreadARDDoubleGradArgs {
    int dim1;
    double *inputX;
    double *precompWeights;
    double *randomFeats;
    double *gradientArray;
    int32_t *sigmaMap;
    double *sigmaVals;
    int startPosition, endPosition;
    int numFreqs;
    int numLengthscales;
    double rbfNormConstant;
};


const char *rbfFeatureGenDouble_(double *cArray, int8_t *radem,
                double *chiArr, double *outputArray,
                double rbfNormConstant,
                int dim0, int dim1, int dim2,
                int numFreqs, int numThreads);

const char *rbfDoubleGrad_(double *cArray, int8_t *radem,
                double *chiArr, double *outputArray,
                double *gradientArray,
                double rbfNormConstant, double sigma,
                int dim0, int dim1, int dim2,
                int numFreqs, int numThreads);

const char *ardDoubleGrad_(double *inputX, double *randomFeatures,
        double *precompWeights, int32_t *sigmaMap, double *sigmaVals,
        double *gradient, int dim0, int dim1, int numLengthscales,
        int numFreqs, double rbfNormConstant, int numThreads);


void *ThreadRBFGenDouble(void *rowArgs);
void *ThreadRBFDoubleGrad(void *rowArgs);
void *ThreadARDDoubleGrad(void *rowArgs);


void rbfDoubleFeatureGenLastStep_(double *xArray, double *chiArray,
        double *outputArray, double normConstant,
        int startRow, int endRow, int dim1,
        int dim2, int numFreqs);

void rbfDoubleGradLastStep_(double *xArray, double *chiArray,
        double *outputArray, double *gradientArray,
        double normConstant, double sigma,
        int startRow, int endRow, int dim1,
        int dim2, int numFreqs);

void ardDoubleGradCalcs_(double *inputX, double *randomFeatures,
        double *precompWeights, int32_t *sigmaMap, double *sigmaVals,
        double *gradient, int startRow, int endRow, int dim1,
        int numLengthscales, double rbfNormConstant, int numFreqs);

#endif
