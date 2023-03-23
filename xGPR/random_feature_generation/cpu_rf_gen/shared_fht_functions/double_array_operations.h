#ifndef DOUBLE_ARRAY_OPERATIONS_H
#define DOUBLE_ARRAY_OPERATIONS_H


void doubleTransformRows3D(double *xArray, int startRow, int endRow,
                    int dim1, int dim2);

void doubleTransformRows2D(double *xArray, int startRow, int endRow,
                    int dim1);

void doubleMultiplyByDiagonalRademacherMat2D(double *xArray,
                    const int8_t *rademArray,
                    int dim1,
                    int startRow, int endRow);

void doubleMultiplyByDiagonalRademacherMat(double *xArray,
                    const int8_t *rademArray,
                    int dim1, int dim2,
                    int startRow, int endRow);

void doubleConv1dMultiplyByRadem(double *xArray,
                        const int8_t *rademArray, int startRow,
                        int endRow, int reshapedDim1,
                        int reshapedDim2, int startPosition);

void doubleConv1dRademAndCopy(double *xArray,
                        double *copyBuffer,
                        const int8_t *rademArray, int startRow,
                        int endRow, int reshapedDim1,
                        int reshapedDim2, int startPosition);
#endif
