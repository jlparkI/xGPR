/*!
 * # shared_rfgen_ops.cpp
 *
 * This module performs core random feature generation
 * operations used by multiple routines.
 */

#include <math.h>
#include <cstring>
#include "shared_rfgen_ops.h"
#include "hadamard_transforms.h"


/*!
 * # multiplyByDiagonalRademacherMat2D
 *
 * Multiplies an input 2d array xArray by a 1d array rademArray assumed
 * to represent a diagonal matrix. rademArray should
 * therefore be of shape (C) if xArray is of shape (N, C).
 * Thus each element (i, j) of xArray is multiplied by
 * element (j) of rademArray. Function assumes caller has
 * verified all dimensions. The array is also multiplied by the normalization
 * constant for the Hadamard transform.
 *
 * ## Args:
 *
 * + `xArray` Pointer to the first element of the array to be
 * modified. Must be a 2d array (e.g. N x C)
 * + `rademArray` A 1d array to multiply against xArray
 * of shape (C)
 * + `dim1` The length of dim2 of xArray (e.g. C in
 * N x C)
 * + `startRow` The first row to modify. When multithreading,
 * the array is split into blocks such that each thread
 * modifies its own subset of the rows.
 * + `endRow` The last row to modify.
 *
 * ## Returns:
 * Operations are in place so nothing is returned.
 */
template <typename T>
void multiplyByDiagonalRademacherMat2D(T __restrict xArray[],
                    const int8_t *rademArray,
                    int dim1,
                    int startRow, int endRow){
    
    int i = startRow, j = i;
    T normConstant = log2(dim1) / 2;
    normConstant = 1 / pow(2, normConstant);
    int rowStride = dim1;
    T *__restrict xElement;
    
    for(i = startRow; i < endRow; i++){
        xElement = xArray + i * rowStride;
        #pragma omp simd
        for (j = 0; j < rowStride; j++){
            *xElement *= rademArray[j] * normConstant;
            xElement++;
        }
    }
}
//Explicitly instantiate for external use.
template void multiplyByDiagonalRademacherMat2D<float>(float *__restrict xArray,
                    const int8_t *rademArray,
                    int dim1,
                    int startRow, int endRow);
template void multiplyByDiagonalRademacherMat2D<double>(double *__restrict xArray,
                    const int8_t *rademArray,
                    int dim1,
                    int startRow, int endRow);





/*!
 * # multiplyByDiagonalRademacherMat
 *
 * Multiplies an input 3d array xArray by a 3d array rademArray assumed
 * to represent a stack of diagonal matrices. rademArray should
 * therefore be of shape (3, D, C) if xArray is of shape (N, D, C).
 * Thus each element (i, j, k) of xArray is multiplied by
 * element (start, j, k) of rademArray. Function assumes caller has
 * verified all dimensions. The array is also multiplied by the normalization
 * constant for the Hadamard transform.
 *
 * ## Args:
 *
 * + `xArray` Pointer to the first element of the array to be
 * modified. Must be a 3d array (e.g. N x D x C)
 * + `rademArray` A 3d array to multiply against xArray
 * of shape (1, D, C)
 * + `dim1` The length of dim2 of xArray (e.g. D in
 * N x D x C)
 * + `dim2` The length of dim3 of xArray (e.g. C in
 * N x D x C)
 * + `startRow` The first row to modify. When multithreading,
 * the array is split into blocks such that each thread
 * modifies its own subset of the rows.
 * + `endRow` The last row to modify.
 *
 * ## Returns:
 * Operations are in place so nothing is returned.
 */
template <typename T>
void multiplyByDiagonalRademacherMat(T __restrict xArray[],
                    const int8_t *rademArray,
                    int dim1, int dim2,
                    int startRow, int endRow){
    
    int i = startRow, j = i;
    T normConstant = log2(dim2) / 2;
    normConstant = 1 / pow(2, normConstant);
    int rowStride = dim1 * dim2;
    T *__restrict xElement;
    
    for(i = startRow; i < endRow; i++){
        xElement = xArray + i * rowStride;
        #pragma omp simd
        for (j = 0; j < rowStride; j++){
            *xElement *= rademArray[j] * normConstant;
            xElement++;
        }
    }
}
//Explicitly instantiate for external use.
template void multiplyByDiagonalRademacherMat<double>(double *__restrict xArray,
                    const int8_t *rademArray,
                    int dim1, int dim2,
                    int startRow, int endRow);
template void multiplyByDiagonalRademacherMat<float>(float *__restrict xArray,
                    const int8_t *rademArray,
                    int dim1, int dim2,
                    int startRow, int endRow);



/*!
 * # singleVectorSORF
 *
 * Performs all the steps involved in the SORF convolution
 * operation on a 1d array.
 *
 * ## Args:
 *
 * + `cbuffer` Pointer to the first element of the 1d array. Size must
 * be a power of 2.
 * + `rademArray` Pointer to the first element of the diagonal rademacher
 * array (size (3,1,F) where F is a multiple of C).
 * + `repeatPosition` A multiple of C that indicates how far along dim2 of
 * rademArray to start.
 * + `rademShape2` dim2 of radem (i.e. F from above).
 * + `cbufferDim2` The size of cbuffer. Must be a power of 2.
 */
template <typename T>
void singleVectorSORF(T cbuffer[], const int8_t *rademArray,
        int repeatPosition, int rademShape2,
        int cbufferDim2){
    T normConstant = log2(cbufferDim2) / 2;
    normConstant = 1 / pow(2, normConstant);
    const int8_t *rademElement = rademArray + repeatPosition;

    #pragma omp simd
    for (int i = 0; i < cbufferDim2; i++)
        cbuffer[i] *= rademElement[i] * normConstant;

    rademElement += rademShape2;
    singleVectorTransform<T>(cbuffer, cbufferDim2);

    #pragma omp simd
    for (int i = 0; i < cbufferDim2; i++)
        cbuffer[i] *= rademElement[i] * normConstant;

    rademElement += rademShape2;
    singleVectorTransform<T>(cbuffer, cbufferDim2);


    #pragma omp simd
    for (int i = 0; i < cbufferDim2; i++)
        cbuffer[i] *= rademElement[i] * normConstant;

    singleVectorTransform<T>(cbuffer, cbufferDim2);
}
//Explicitly instantiate for external use.
template void singleVectorSORF<double>(double cbuffer[], const int8_t *rademArray,
        int repeatPosition, int rademShape2,
        int cbufferDim2);
template void singleVectorSORF<float>(float cbuffer[], const int8_t *rademArray,
        int repeatPosition, int rademShape2,
        int cbufferDim2);





/*!
 * # singleVectorRBFPostProcess
 *
 * Performs the last steps in RBF-based kernel feature
 * generation for a single vector. This can be used either
 * for convolution or standard RBF kernels.
 *
 * ## Args:
 * + `xdata` Pointer to the first element of the array that has been
 * used for the convolution. Shape is (C). C must be a power of 2.
 * + `chiArr` Pointer to the first element of chiArr, a diagonal array
 * that will be multipled against xdata.
 * + `outputArray` A pointer to the first element of the array in which
 * the output will be stored.
 * + `dim2` The last dimension of xdata
 * + `numFreqs` The number of frequencies to sample.
 * + `rowNumber` The row of the output array to use.
 * + `repeatNum` The repeat number
 * + `convWidth` The convolution width
 * + `scalingTerm` The scaling term to apply for the random feature generation.
 *
 */
template <typename T>
void singleVectorRBFPostProcess(const T xdata[],
        const T chiArr[], double *outputArray,
        int dim2, int numFreqs,
        int rowNumber, int repeatNum,
        double scalingTerm){

    int outputStart = repeatNum * dim2;
    T prodVal;
    double *__restrict xOut;
    const T *chiIn;
    //NOTE: MIN is defined in the header.
    int endPosition = MIN(numFreqs, (repeatNum + 1) * dim2);
    endPosition -= outputStart;

    chiIn = chiArr + outputStart;
    xOut = outputArray + 2 * outputStart + rowNumber * 2 * numFreqs;

    #pragma omp simd
    for (int i=0; i < endPosition; i++){
        prodVal = xdata[i] * chiIn[i];
        *xOut += cos(prodVal) * scalingTerm;
        xOut++;
        *xOut += sin(prodVal) * scalingTerm;
        xOut++;
    }
}
//Explicitly instantiate for external use.
template void singleVectorRBFPostProcess<double>(const double xdata[], const double chiArr[],
        double *outputArray, int dim2, int numFreqs, int rowNumber, int repeatNum,
        double scalingTerm);
template void singleVectorRBFPostProcess<float>(const float xdata[], const float chiArr[],
        double *outputArray, int dim2, int numFreqs, int rowNumber, int repeatNum,
        double scalingTerm);



/*!
 * # singleVectorRBFPostGrad
 *
 * Performs the last steps in RBF-based kernel feature
 * generation for a single vector. This can be used
 * either for convolution or standard RBF kernels.
 *
 * ## Args:
 * + `xdata` Pointer to the first element of the array that has been
 * used for the convolution. Shape is (C). C must be a power of 2.
 * + `chiArr` Pointer to the first element of chiArr, a diagonal array
 * that will be multipled against xdata.
 * + `outputArray` A pointer to the first element of the array in which
 * the output will be stored.
 * + `gradientArray` A pointer to the first element of the array in
 * which the gradient will be stored.
 * + `sigma` The sigma hyperparameter.
 * + `dim2` The last dimension of xdata
 * + `numFreqs` The number of frequencies to sample.
 * + `rowNumber` The row of the output array to use.
 * + `repeatNum` The repeat number
 * + `convWidth` The convolution width
 * + `scalingTerm` The scaling term to apply for the random feature generation.
 *
 */
template <typename T>
void singleVectorRBFPostGrad(const T xdata[],
        const T chiArr[], double *outputArray,
        double *gradientArray, T sigma,
        int dim2, int numFreqs,
        int rowNumber, int repeatNum,
        double scalingTerm){

    int outputStart = repeatNum * dim2;
    T prodVal, gradVal, cosVal, sinVal;
    double *__restrict xOut, *__restrict gradOut;
    const T *chiIn;
    //NOTE: MIN is defined in the header.
    int endPosition = MIN(numFreqs, (repeatNum + 1) * dim2);
    endPosition -= outputStart;

    chiIn = chiArr + outputStart;
    xOut = outputArray + 2 * outputStart + rowNumber * 2 * numFreqs;
    gradOut = gradientArray + 2 * outputStart + rowNumber * 2 * numFreqs;

    for (int i=0; i < endPosition; i++){
        gradVal = xdata[i] * chiIn[i];
        prodVal = gradVal * sigma;
        cosVal = cos(prodVal) * scalingTerm;
        sinVal = sin(prodVal) * scalingTerm;
        *xOut += cosVal;
        xOut++;
        *xOut += sinVal;
        xOut++;
        *gradOut -= sinVal * gradVal;
        gradOut++;
        *gradOut += cosVal * gradVal;
        gradOut++;
    }
}
//Explicitly instantiate for external use.
template void singleVectorRBFPostGrad<double>(const double xdata[], const double chiArr[],
        double *outputArray, double *gradientArray, double sigma,
        int dim2, int numFreqs, int rowNumber, int repeatNum,
        double scalingTerm);
template void singleVectorRBFPostGrad<float>(const float xdata[], const float chiArr[],
        double *outputArray, double *gradientArray, float sigma,
        int dim2, int numFreqs, int rowNumber, int repeatNum,
        double scalingTerm);
