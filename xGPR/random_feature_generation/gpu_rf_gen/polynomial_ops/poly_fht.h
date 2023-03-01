#ifndef CUDA_POLY_FHT_PREP_H
#define CUDA_POLY_FHT_PREP_H

const char *floatPolyConvFHTPrep(int8_t *radem, float *reshapedX, int reshapedDim0,
                int reshapedDim1, int reshapedDim2, int numFreqs,
                int columnStartPosition, int rowStartPosition);

const char *doublePolyConvFHTPrep(int8_t *radem, double *reshapedX, int reshapedDim0,
                int reshapedDim1, int reshapedDim2, int numFreqs,
                int columnStartPosition, int rowStartPosition);


const char *floatPolyFHTPrep(int8_t *radem, float *reshapedX, int reshapedDim0, 
                int reshapedDim1, int reshapedDim2, int rademStartPosition);

const char *doublePolyFHTPrep(int8_t *radem, double *reshapedX, int reshapedDim0, 
                int reshapedDim1, int reshapedDim2, int rademStartPosition);

#endif
