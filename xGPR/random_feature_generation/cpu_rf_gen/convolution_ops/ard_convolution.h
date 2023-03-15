#ifndef ARD_CONVOLUTION_H
#define ARD_CONVOLUTION_H


struct ThreadGraphARDDoubleGradArgs {
    int dim1, dim2;
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


struct ThreadGraphARDFloatGradArgs {
    int dim1, dim2;
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


const char *graphARDDoubleGrad_(double *inputX, double *randomFeatures,
        double *precompWeights, int32_t *sigmaMap, double *sigmaVals,
        double *gradient, int dim0, int dim1, int dim2,
        int numLengthscales, int numFreqs, double rbfNormConstant,
        int numThreads);


const char *graphARDFloatGrad_(float *inputX, double *randomFeatures,
        float *precompWeights, int32_t *sigmaMap, double *sigmaVals,
        double *gradient, int dim0, int dim1, int dim2,
        int numLengthscales, int numFreqs, double rbfNormConstant,
        int numThreads);

void *DoubleThreadGraphARDGrad(void *sharedArgs);
void *FloatThreadGraphARDGrad(void *sharedArgs);


void doubleGraphARDGradCalcs(double *inputX, double *randomFeatures,
        double *precompWeights, int32_t *sigmaMap, double *sigmaVals,
        double *gradient, int startRow, int endRow, int dim1,
        int dim2, int numLengthscales, double rbfNormConstant,
        int numFreqs);

void floatGraphARDGradCalcs(float *inputX, double *randomFeatures,
        float *precompWeights, int32_t *sigmaMap, double *sigmaVals,
        double *gradient, int startRow, int endRow, int dim1,
        int dim2, int numLengthscales, double rbfNormConstant,
        int numFreqs);

#endif
