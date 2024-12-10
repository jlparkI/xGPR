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
void singleVectorSORF(T cbuffer[], const int8_t *rademArray,
        int repeatPosition, int rademShape2,
        int cbufferDim2);

template <typename T>
void singleVectorRBFPostProcess(const T xdata[],
        const T chiArr[], double *outputArray,
        int dim2, int numFreqs,
        int rowNumber, int repeatNum,
        double scalingTerm);

template <typename T>
void singleVectorRBFPostGrad(const T xdata[],
        const T chiArr[], double *outputArray,
        double *gradientArray, T sigma,
        int dim2, int numFreqs,
        int rowNumber, int repeatNum,
        double scalingTerm);

#endif
