#ifndef DOUBLE_CUDA_SPECIALIZED_OPERATIONS_H
#define DOUBLE_CUDA_SPECIALIZED_OPERATIONS_H

const char *doubleRBFFeatureGen(double *cArray, int8_t *radem,
                double *chiArr, double *outputArray,
                double rbfNormConstant,
                int dim0, int dim1, int dim2,
                int numFreqs);

#endif
