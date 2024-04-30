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
#include "simplex_rff_projections.h"


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
 * # multiplyByDiagonalRademAndCopy
 *
 * Multiplies an input 3d array xArray by a 3d array rademArray assumed
 * to represent a stack of diagonal matrices. rademArray should
 * therefore be of shape (3, D, C) if xArray is of shape (N, D, C).
 * Thus each element (i, j, k) of xArray is multiplied by
 * element (start, j, k) of rademArray. Function assumes caller has
 * verified all dimensions. The array is also multiplied by the normalization
 * constant for the Hadamard transform, and is copied to the output
 * array copyBuffer in which results are stored; no other input array
 * is modified.
 *
 * ## Args:
 *
 * + `xArray` Pointer to the first element of the array to be
 * multiplied. Must be a 3d array (e.g. N x D x C)
 * + `copyBuffer` Pointer to the first element of the output
 * array into which the results will be written. Must be
 * same size / shape as xArray.
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
void multiplyByDiagonalRademAndCopy(const T xArray[],
                    T copyBuffer[],
                    const int8_t *rademArray,
                    int dim1, int dim2,
                    int startRow, int endRow){
    
    int i = startRow, j = i;
    T normConstant = log2(dim2) / 2;
    normConstant = 1 / pow(2, normConstant);
    int rowStride = dim1 * dim2;
    const T *xElement;
    T *__restrict outElement;
    
    for(i = startRow; i < endRow; i++){
        xElement = xArray + i * rowStride;
        outElement = copyBuffer + i * rowStride;
        #pragma omp simd
        for (j = 0; j < rowStride; j++){
            *outElement = *xElement * rademArray[j] * normConstant;
            xElement++;
            outElement++;
        }
    }
}
//Explicitly instantiate for external use.
template void multiplyByDiagonalRademAndCopy<double>(const double *xArray,
                    double *copyBuffer,
                    const int8_t *rademArray,
                    int dim1, int dim2,
                    int startRow, int endRow);
template void multiplyByDiagonalRademAndCopy<float>(const float *xArray,
                    float *copyBuffer,
                    const int8_t *rademArray,
                    int dim1, int dim2,
                    int startRow, int endRow);


/*!
 * # conv1dMultiplyByRadem
 *
 * Multiplies an input 3d array xArray by a 3d array rademArray assumed
 * to represent a stack of diagonal matrices. rademArray should
 * be of shape (3, 1, C * m) if xArray is of shape (N, D, C)
 * where m is an integer corresponding to the number of blocks
 * of random features that need to be generated.
 * Thus each element (i, j, k) of xArray is multiplied by
 * element (start, 0, p * C + k) of rademArray where p is an iterator
 * that is increased while p < m. Function assumes caller has
 * verified all dimensions. The array is also multiplied by
 * the normalization constant for the Hadamard transform.
 *
 * ## Args:
 *
 * + `xArray` Pointer to the first element of the array to be
 * modified. Must be a 3d array (e.g. N x D x C)
 * + `rademArray` A 3d array to multiply against xArray
 * of shape (m, 1, F) where F is a multiple of C.
 * + `reshapedDim1` The length of dim2 of xArray (e.g. D in
 * N x D x C)
 * + `reshapedDim2` The length of dim3 of xArray (e.g. C in
 * N x D x C)
 * + `startRow` The first row to modify. When multithreading,
 * the array is split into blocks such that each thread
 * modifies its own subset of the rows.
 * + `endRow` The last row to modify.
 * + `startPosition` Where to start in radem.
 * 
 * ## Returns:
 * Operations are in place so nothing is returned.
 */
template <typename T>
void conv1dMultiplyByRadem(T __restrict xArray[],
                        const int8_t *rademArray, int startRow,
                        int endRow, int reshapedDim1,
                        int reshapedDim2, int startPosition){
    int j, k;
    T normConstant = log2(reshapedDim2) / 2;
    normConstant = 1 / pow(2, normConstant);
    int rowStride = reshapedDim1 * reshapedDim2;
    T *__restrict xElement;

    for (int i = startRow; i < endRow; i++){
        xElement = xArray + i * rowStride;
        #pragma omp simd
        for (j = 0; j < reshapedDim1; j++){
            for (k = 0; k < reshapedDim2; k++){
                *xElement *= rademArray[startPosition + k] * normConstant;
                xElement++;
            }
        }
    }
}
//Explicitly instantiate for external use.
template void conv1dMultiplyByRadem<double>(double *__restrict xArray,
                        const int8_t *rademArray, int startRow,
                        int endRow, int reshapedDim1,
                        int reshapedDim2, int startPosition);
template void conv1dMultiplyByRadem<float>(float *__restrict xArray,
                        const int8_t *rademArray, int startRow,
                        int endRow, int reshapedDim1,
                        int reshapedDim2, int startPosition);



/*!
 * #StridedCopyOp
 *
 * Copies from a raw data array into a copy buffer in such a way that
 * an FHT-based convolution can be performed on the copy buffer. Caller
 * must check parameters to ensure they are valid.
 *
 * ## Args:
 *
 * + `xdata` Pointer to the first element of the raw data array.
 * + `copyBuffer` Pointer to the first element of the copy buffer.
 * + `dim1` shape[1] of xdata
 * + `dim2` shape[2] of xdata
 * + `bufferDim2` shape[2] of copyBuffer.
 * + `startRow` first row of xdata to copy
 * + `endRow` last row of xdata to copy
 * + `convWidth` The number of elements in a convolution.
 *
 */
template <typename T>
void StridedCopyOp(const T xdata[], T copyBuffer[],
        int dim1, int dim2, int bufferDim2, int startRow, int endRow,
        int convWidth){
    int inputRowSize = dim1 * dim2;
    int numKmers = dim1 - convWidth + 1;
    
    int inputElementSize = dim2 * convWidth;
    int outputRowSize = bufferDim2 * numKmers;

    const T *__restrict xPtr = xdata + startRow * inputRowSize;
    T *__restrict copyPtr = copyBuffer + startRow * outputRowSize;

    for (int i=startRow; i < endRow; i++){
        xPtr = xdata + i * inputRowSize;
        copyPtr = copyBuffer + i * outputRowSize;

        for (int j=0; j < numKmers; j++){
            for (int k=0; k < inputElementSize; k++)
                copyPtr[k] = xPtr[k];
            for (int k=inputElementSize; k < bufferDim2; k++)
                copyPtr[k] = 0;

            copyPtr += bufferDim2;
            xPtr += dim2;
        }
    }
}
template void StridedCopyOp<double>(const double *xdata,
                        double *copyBuffer, int dim1, int dim2,
                        int bufferDim2, int startRow, int endRow,
                        int convWidth);
template void StridedCopyOp<float>(const float *xdata,
                        float *copyBuffer, int dim1, int dim2,
                        int bufferDim2, int startRow, int endRow,
                        int convWidth);



/*!
 * # SORF3D
 *
 * Performs all the steps involved in the SORF operation on
 * a 3d array. Can be used with a 2d array by passing dim1=1.
 *
 * ## Args:
 *
 * + `xArray` Pointer to the first element of the array to be
 * modified. Must be a 3d array (e.g. N x D x C). C MUST be
 * a power of 2.
 * + `rademArray` Pointer to the first element of the array containing
 * the diagonal Rademacher matrix (size (m,D,C)).
 * + `startPosition` The first row to modify. When multithreading,
 * the array is split into blocks such that each thread
 * modifies its own subset of the rows.
 * + `endRow` The last row to modify.
 * + `dim1` The length of dim2 of the array (e.g. D in
 * N x D x C)
 * + `dim2` The length of dim3 of the array (e.g. C in
 * N x D x C)
 */
template <typename T>
void SORF3D(T xArray[], const int8_t *rademArray,
        int startPosition, int endPosition, int dim1, int dim2){
    int rowSize = dim1 * dim2;

    multiplyByDiagonalRademacherMat<T>(xArray,
                    rademArray,
                    dim1, dim2, 
                    startPosition, endPosition);
    transformRows<T>(xArray, startPosition, 
                    endPosition, dim1, dim2);

    multiplyByDiagonalRademacherMat<T>(xArray,
                    rademArray + rowSize,
                    dim1, dim2, 
                    startPosition, endPosition);
    transformRows<T>(xArray, startPosition, 
                    endPosition, dim1, dim2);
    
    multiplyByDiagonalRademacherMat<T>(xArray,
                    rademArray + 2 * rowSize,
                    dim1, dim2, 
                    startPosition, endPosition);
    transformRows<T>(xArray, startPosition, 
                    endPosition, dim1, dim2);
}
//Explicitly instantiate for external use.
template void SORF3D<double>(double *xArray,
                        const int8_t *rademArray, int startPosition,
                        int endPosition, int dim1, int dim2);
template void SORF3D<float>(float *xArray,
                        const int8_t *rademArray, int startPosition,
                        int endPosition, int dim1, int dim2);



/*!
 * # convSORF3D
 *
 * Performs all the steps involved in the SORF convolution
 * operation on a 3d array. Can be used with a 2d array by
 * passing dim1=1.
 *
 * ## Args:
 *
 * + `reshapedXArray` Pointer to the first element of the array from
 * which data will be copied. Must be a 3d array (e.g. N x D x C). C MUST be
 * a power of 2.
 * + `rademArray` Pointer to the first element of the diagonal rademacher
 * array (size (3,1,F) where F is a multiple of C).
 * + `repeatPosition` A multiple of C that indicates how far along dim2 of
 * rademArray to start.
 * + `startRow` The first row to modify. When multithreading,
 * the array is split into blocks such that each thread
 * modifies its own subset of the rows.
 * + `endRow` The last row to modify.
 * + `dim1` The length of dim2 of the array (e.g. D in
 * N x D x C)
 * + `dim2` The length of dim3 of the array (e.g. C in
 * N x D x C)
 * + `rademShape2` The size of F in the shape of the rademArray.
 */
template <typename T>
void convSORF3D(T xArray[], const int8_t *rademArray,
        int repeatPosition, int startRow, int endRow,
        int dim1, int dim2, int rademShape2){

    conv1dMultiplyByRadem<T>(xArray,
                    rademArray,
                    startRow, endRow,
                    dim1, dim2,
                    repeatPosition);
    transformRows<T>(xArray, startRow,
                    endRow, dim1, dim2);

    conv1dMultiplyByRadem<T>(xArray,
                    rademArray + rademShape2,
                    startRow, endRow,
                    dim1, dim2,
                    repeatPosition);
    transformRows<T>(xArray, startRow,
                    endRow, dim1, dim2);

    conv1dMultiplyByRadem<T>(xArray,
                    rademArray + 2 * rademShape2,
                    startRow, endRow,
                    dim1, dim2,
                    repeatPosition);
    transformRows<T>(xArray, startRow,
                    endRow, dim1, dim2);
}
//Explicitly instantiate for external use.
template void convSORF3D<double>(double *xArray,
                        const int8_t *rademArray, int repeatPosition,
                        int startRow, int endRow, int dim1, int dim2,
                        int rademShape2);
template void convSORF3D<float>(float *xArray,
                        const int8_t *rademArray, int repeatPosition,
                        int startRow, int endRow, int dim1, int dim2,
                        int rademShape2);



/*!
 * # convSORF3DWithCopyBuffer
 *
 * Performs all the steps involved in the SORF convolution
 * operation on a 3d array, but with a copy buffer into which the data
 * is copied before any operations are performed. Can be used with a 2d
 * array by passing dim1=1.
 *
 * ## Args:
 *
 * + `reshapedXArray` Pointer to the first element of the array from
 * which data will be copied. Must be a 3d array (e.g. N x D x C). C MUST be
 * a power of 2.
 * + `copyBuffer` Pointer to the first element of copyBuffer, which must
 * be the same size as reshapedXArray.
 * + `rademArray` Pointer to the first element of the diagonal rademacher
 * array (size (3,1,F) where F is a multiple of C).
 * + `repeatPosition` A multiple of C that indicates how far along dim2 of
 * rademArray to start.
 * + `startRow` The first row to modify. When multithreading,
 * the array is split into blocks such that each thread
 * modifies its own subset of the rows.
 * + `endRow` The last row to modify.
 * + `dim1` The length of dim2 of the array (e.g. D in
 * N x D x C)
 * + `dim2` The length of dim3 of the array (e.g. C in
 * N x D x C)
 * + `rademShape2` The size of F in the shape of the rademArray.
 * + `convWidth` The number of elements in a convolution.
 * + `bufferDim2` The dim2 of the copy buffer.
 */
template <typename T>
void convSORF3DWithCopyBuffer(T reshapedXArray[], T copyBuffer[],
        const int8_t *rademArray,
        int repeatPosition, int startRow, int endRow,
        int dim1, int dim2, int rademShape2,
        int convWidth, int bufferDim2){

    int numKmers = dim1 - convWidth + 1;

    StridedCopyOp(reshapedXArray, copyBuffer,
            dim1, dim2, bufferDim2, startRow, endRow,
            convWidth);
    conv1dMultiplyByRadem<T>(copyBuffer,
                    rademArray,
                    startRow, endRow,
                    numKmers, bufferDim2,
                    repeatPosition);
    transformRows<T>(copyBuffer, startRow,
                    endRow, numKmers, bufferDim2);

    conv1dMultiplyByRadem<T>(copyBuffer,
                    rademArray + rademShape2,
                    startRow, endRow,
                    numKmers, bufferDim2,
                    repeatPosition);
    transformRows<T>(copyBuffer, startRow,
                    endRow, numKmers, bufferDim2);

    conv1dMultiplyByRadem<T>(copyBuffer,
                    rademArray + 2 * rademShape2,
                    startRow, endRow,
                    numKmers, bufferDim2,
                    repeatPosition);
    transformRows<T>(copyBuffer, startRow,
                    endRow, numKmers, bufferDim2);
}
//Explicitly instantiate for external use.
template void convSORF3DWithCopyBuffer<double>(double *reshapedXArray, double *copyBuffer,
                        const int8_t *rademArray, int repeatPosition,
                        int startRow, int endRow, int dim1, int dim2,
                        int rademShape2, int convWidth, int bufferDim2);
template void convSORF3DWithCopyBuffer<float>(float *reshapedXArray, float *copyBuffer,
                        const int8_t *rademArray, int repeatPosition,
                        int startRow, int endRow, int dim1, int dim2,
                        int rademShape2, int convWidth, int bufferDim2);



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

    singleVectorSimplexProj(cbuffer, cbufferDim2);
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
