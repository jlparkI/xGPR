#ifndef POLY_SRHT_H
#define POLY_SRHT_H

#define VALID_INPUTS 0
#define INVALID_INPUTS 1
#define EPS_TOLERANCE 0.0
#define MAX_THREADS 8

const char *floatPolyFHTPrep_(int8_t *radem, float *reshapedX,
            int numThreads, int reshapedDim0,
            int reshapedDim1, int reshapedDim2,
            int rademStartPosition);

const char *floatPolyConvFHTPrep_(int8_t *radem, float *reshapedX,
            int numThreads, int reshapedDim0,
            int reshapedDim1, int reshapedDim2, int numFreqs,
            int repeatNum, int rademStartPosition);

const char *doublePolyFHTPrep_(int8_t *radem, double *reshapedX,
            int numThreads, int reshapedDim0,
            int reshapedDim1, int reshapedDim2,
            int rademStartPosition);

const char *doublePolyConvFHTPrep_(int8_t *radem, double *reshapedX,
            int numThreads, int reshapedDim0,
            int reshapedDim1, int reshapedDim2, int numFreqs,
            int repeatNum, int rademStartPosition);


void *ThreadPolyFHTFloat(void *sharedArgs);
void *ThreadPolyFHTDouble(void *sharedArgs);

void *ThreadPolyConvFHTFloat(void *sharedArgs);
void *ThreadPolyConvFHTDouble(void *sharedArgs);
#endif
