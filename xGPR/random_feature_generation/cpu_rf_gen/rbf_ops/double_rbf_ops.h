#ifndef DOUBLE_RBF_OPS_H
#define DOUBLE_RBF_OPS_H

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


void *ThreadRBFGenDouble(void *rowArgs);
void *ThreadRBFDoubleGrad(void *rowArgs);


void rbfDoubleFeatureGenLastStep_(double *xArray, double *chiArray,
        double *outputArray, double normConstant,
        int startRow, int endRow, int dim1,
        int dim2, int numFreqs);

void rbfDoubleGradLastStep_(double *xArray, double *chiArray,
        double *outputArray, double *gradientArray,
        double normConstant, double sigma,
        int startRow, int endRow, int dim1,
        int dim2, int numFreqs);

#endif
