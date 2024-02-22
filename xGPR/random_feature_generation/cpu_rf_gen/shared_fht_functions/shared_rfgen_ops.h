#ifndef SHARED_RFGEN_OPERATIONS_H
#define SHARED_RFGEN_OPERATIONS_H
#include <stdint.h>


template <typename T>
void multiplyByDiagonalRademacherMat2D(T __restrict xArray[],
                    const int8_t *rademArray,
                    int dim1,
                    int startRow, int endRow);

template <typename T>
void multiplyByDiagonalRademacherMat(T __restrict xArray[],
                    const int8_t *rademArray,
                    int dim1, int dim2,
                    int startRow, int endRow);

template <typename T>
void multiplyByDiagonalRademAndCopy(const T xArray[],
                    T copyBuffer[],
                    const int8_t *rademArray,
                    int dim1, int dim2,
                    int startRow, int endRow);


template <typename T>
void conv1dMultiplyByRadem(T __restrict xArray[],
                        const int8_t *rademArray, int startRow,
                        int endRow, int reshapedDim1,
                        int reshapedDim2, int startPosition);

template <typename T>
void StridedCopyOp(const T xdata[], T copyBuffer[],
        int dim1, int dim2, int bufferDim2, int startRow,
        int endRow, int convWidth);

template <typename T>
void SORF3D(T arrayStart[], const int8_t *rademArray,
        int startPosition, int endPosition, int dim1, int dim2);

template <typename T>
void convSORF3D(T arrayStart[], const int8_t *rademArray,
        int repeatPosition, int startRow, int endRow,
        int dim1, int dim2, int rademShape2);

template <typename T>
void convSORF3DWithCopyBuffer(T reshapedXArray[], T copyBuffer[],
        const int8_t *rademArray, int repeatPosition, int startRow,
        int endRow, int dim1, int dim2, int rademShape2,
        int convWidth, int bufferDim2);


#endif
