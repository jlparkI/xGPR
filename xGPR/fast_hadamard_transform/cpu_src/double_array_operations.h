#ifndef DOUBLE_ARRAY_OPERATIONS_H
#define DOUBLE_ARRAY_OPERATIONS_H

#include <Python.h>
#include <numpy/arrayobject.h>

void doubleTransformRows3D(double *xArray, int startRow, int endRow,
                    int dim1, int dim2);

void doubleTransformRows2D(double *xArray, int startRow, int endRow,
                    int dim1);

void doubleMultiplyByDiagonalRademacherMat2D(double *xArray,
                    int8_t *rademArray,
                    int dim1,
                    int startRow, int endRow);

void doubleMultiplyByDiagonalRademacherMat(double *xArray,
                    int8_t *rademArray,
                    int dim1, int dim2,
                    int startRow, int endRow);

void doubleConv1dMultiplyByRadem(double *xArray,
                        int8_t *rademArray, int startRow,
                        int endRow, int reshapedDim1,
                        int reshapedDim2, int startPosition);
#endif
