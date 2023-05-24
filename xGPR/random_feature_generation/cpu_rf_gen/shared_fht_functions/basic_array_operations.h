#ifndef BASIC_ARRAY_OPERATIONS_H
#define BASIC_ARRAY_OPERATIONS_H


template <typename T>
void transformRows3D(T xArray[], int startRow, int endRow,
                    int dim1, int dim2);

template <typename T>
void transformRows2D(T xArray[], int startRow, int endRow,
                    int dim1);

template <typename T>
void multiplyByDiagonalRademacherMat2D(T xArray[],
                    const int8_t *rademArray,
                    int dim1,
                    int startRow, int endRow);

template <typename T>
void multiplyByDiagonalRademacherMat(T xArray[],
                    const int8_t *rademArray,
                    int dim1, int dim2,
                    int startRow, int endRow);

template <typename T>
void conv1dMultiplyByRadem(T xArray[],
                        const int8_t *rademArray, int startRow,
                        int endRow, int reshapedDim1,
                        int reshapedDim2, int startPosition);

template <typename T>
void conv1dRademAndCopy(T xArray[],
                        T copyBuffer[],
                        const int8_t *rademArray, int startRow,
                        int endRow, int reshapedDim1,
                        int reshapedDim2, int startPosition);
#endif
