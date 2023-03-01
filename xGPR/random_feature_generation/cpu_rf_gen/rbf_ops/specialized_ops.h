#ifndef SPECIALIZED_OPS_H
#define SPECIALIZED_OPS_H

#include <Python.h>
#include <numpy/arrayobject.h>

#define VALID_INPUTS 0
#define INVALID_INPUTS 1
#define EPS_TOLERANCE 0.0
#define MAX_THREADS 8

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

const char *rbfFeatureGenFloat_(float *cArray, int8_t *radem,
                float *chiArr, double *outputArray,
                double rbfNormConstant,
                int dim0, int dim1, int dim2,
                int numFreqs, int numThreads);

const char *rbfFeatureGenDouble_(double *cArray, int8_t *radem,
                double *chiArr, double *outputArray,
                double rbfNormConstant,
                int dim0, int dim1, int dim2,
                int numFreqs, int numThreads);


void *ThreadRBFGenFloat(void *rowArgs);
void *ThreadRBFGenDouble(void *rowArgs);


void rbfFloatFeatureGenLastStep_(float *xArray, float *chiArray,
        double *outputArray, double normConstant,
        int startRow, int endRow, int dim1,
        int dim2, int numFreqs);
void rbfDoubleFeatureGenLastStep_(double *xArray, double *chiArray,
        double *outputArray, double normConstant,
        int startRow, int endRow, int dim1,
        int dim2, int numFreqs);

#endif
