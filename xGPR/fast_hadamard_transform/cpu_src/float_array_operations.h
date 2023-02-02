#ifndef FLOAT_ARRAY_OPERATIONS_H
#define FLOAT_ARRAY_OPERATIONS_H

#include <Python.h>
#include <numpy/arrayobject.h>

void floatTransformRows3D(float *xArray, int startRow, int endRow,
                    int dim1, int dim2);

void floatTransformRows2D(float *xArray, int startRow, int endRow,
                    int dim1);

void floatMultiplyByDiagonalRademacherMat2D(float *xArray,
                    int8_t *rademArray,
                    int dim1,
                    int startRow, int endRow);

void floatMultiplyByDiagonalRademacherMat(float *xArray,
                    int8_t *rademArray,
                    int dim1, int dim2,
                    int startRow, int endRow);

void floatMultiplyByDiagonalMat2D(float *xArray,
                    float *diagArray,
                    int dim1,
                    int startRow, int endRow);

void floatMultiplyByDiagonalMat(float *xArray,
                    float *diagArray,
                    int dim1, int dim2,
                    int startRow, int endRow);

void floatConv1dMultiplyByRadem(float *xArray,
                        int8_t *rademArray, int startRow,
                        int endRow, int reshapedDim1,
                        int reshapedDim2, int startPosition);

void floatConv1dMultiplyByDiagonalMat(float *xArray,
                        float *diagArray, int startRow,
                        int endRow, int reshapedDim1,
                        int reshapedDim2, int startPosition);

#endif
