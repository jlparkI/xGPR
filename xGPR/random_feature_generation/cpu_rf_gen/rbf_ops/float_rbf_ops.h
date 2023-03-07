#ifndef FLOAT_RBF_OPS_H
#define FLOAT_RBF_OPS_H

#include <Python.h>
#include <numpy/arrayobject.h>

#define VALID_INPUTS 0
#define INVALID_INPUTS 1
#define EPS_TOLERANCE 0.0
#define MAX_THREADS 8

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


void *ThreadRBFGenFloat(void *rowArgs);
void *ThreadRBFFloatGrad(void *rowArgs);


void rbfFloatFeatureGenLastStep_(float *xArray, float *chiArray,
        double *outputArray, double normConstant,
        int startRow, int endRow, int dim1,
        int dim2, int numFreqs);

void rbfFloatGradLastStep_(float *xArray, float *chiArray,
        double *outputArray, double *gradientArray,
        double normConstant, float sigma,
        int startRow, int endRow, int dim1,
        int dim2, int numFreqs);

#endif
