/*!
 * # float_array_operations.c
 *
 * This module performs core Hadamard transform and diagonal matrix
 * multiplication operations when the input is an array of floats.
 * It includes the following functions:
 *
 * + floatTransformRows3D
 * Performs the unnormalized Hadamard transform on a 3d array
 *
 * + floatTransformRows2D
 * Performs the unnormalized Hadamard transform on a 2d array
 *
 * + floatMultiplyByDiagonalRademacherMat2D
 * Multiplies a 2d array by a diagonal matrix whose elements are
 * drawn from a Rademacher distribution
 *
 * + floatMultiplyByDiagonalRademacherMat
 * Multiplies a 3d array by a diagonal matrix whose elements are
 * drawn from a Rademacher distribution
 *
 * + floatConv1dMultiplyByRadem
 * Same as floatMultiplyByDiagonalRademacherMat, but designed to work
 * on 3d arrays structured to perform FHT-based convolution.
 */

#include "float_array_operations.h"


/*!
 * # floatTransformRows3D
 *
 * Performs an unnormalized Hadamard transform along the last
 * dimension of an input 3d array. The transform is performed
 * in place so nothing is returned. Assumes dimensions have
 * already been checked by caller. Designed to be compatible
 * with multithreading.
 *
 * ## Args:
 *
 * + `xArray` Pointer to the first element of the array to be
 * modified. Must be a 3d array (e.g. N x D x C). C MUST be
 * a power of 2.
 * + `startRow` The first row to modify. When multithreading,
 * the array is split into blocks such that each thread
 * modifies its own subset of the rows.
 * + `endRow` The last row to modify.
 * + `dim1` The length of dim2 of the array (e.g. D in
 * N x D x C)
 * + `dim2` The length of dim3 of the array (e.g. C in
 * N x D x C)
 */
void floatTransformRows3D(float *xArray, int startRow, int endRow,
                    int dim1, int dim2){
    int idx1 = startRow;
    int i = 0, j, h = 1;
    float y;
    int rowStride = dim1 * dim2;
    float *xElement, *yElement;

    //Unrolling the first few loops
    //of the transform increased speed substantially
    //(may be compiler and optimization level dependent).
    //This yields diminishing returns and does not
    //offer much improvement past 3 unrolls.
    for (idx1 = startRow; idx1 < endRow; idx1++){
        xElement = xArray + idx1 * rowStride;
        yElement = xElement + 1;
        for (i = 0; i < rowStride; i += 2){
            y = *yElement;
            *yElement = *xElement - y;
            *xElement += y;
            xElement += 2;
            yElement += 2;
        }
        if (dim2 <= 2)
            continue;
        
        xElement = xArray + idx1 * rowStride;
        yElement = xElement + 2;
	    for (i = 0; i < rowStride; i += 4){
            y = *yElement;
            *yElement = *xElement - y;
            *xElement += y;
            xElement ++;
            yElement ++;
            y = *yElement;
            *yElement = *xElement - y;
            *xElement += y;
            xElement += 3;
            yElement += 3;
        }
        if (dim2 <= 4)
            continue;

        xElement = xArray + idx1 * rowStride;
        yElement = xElement + 4;
	    for (i = 0; i < rowStride; i += 8){
            y = *yElement;
            *yElement = *xElement - y;
            *xElement += y;
            xElement ++;
            yElement ++;
            
            y = *yElement;
            *yElement = *xElement - y;
            *xElement += y;
            xElement ++;
            yElement ++;
            
            y = *yElement;
            *yElement = *xElement - y;
            *xElement += y;
            xElement ++;
            yElement ++;
            y = *yElement;
            *yElement = *xElement - y;
            *xElement += y;
            
            xElement += 5;
            yElement += 5;
        }
        if (dim2 <= 8)
            continue;

        //The general, non-unrolled transform.
        for (h = 8; h < dim2; h <<= 1){
            for (i = 0; i < rowStride; i += (h << 1)){
                xElement = xArray + idx1 * rowStride + i;
                yElement = xElement + h;
                for (j=0; j < h; j++){
                    y = *yElement;
                    *yElement = *xElement - y;
                    *xElement += y;
                    xElement++;
                    yElement++;
                }
            }
        }
    }
}



/*!
 * # floatTransformRows2D
 *
 * Performs an unnormalized Hadamard transform along the last
 * dimension of an input 2d array. The transform is performed
 * in place so nothing is returned. Assumes dimensions have
 * already been checked by caller. Designed to be compatible
 * with multithreading.
 *
 * ## Args:
 *
 * + `xArray` Pointer to the first element of the array to be
 * modified. Must be a 2d array (e.g. N x C). C MUST be
 * a power of 2.
 * + `startRow` The first row to modify. When multithreading,
 * the array is split into blocks such that each thread
 * modifies its own subset of the rows.
 * + `endRow` The last row to modify.
 * + `dim1` The length of dim2 of the array (e.g. C in
 * N x C)
 */
void floatTransformRows2D(float *xArray, int startRow, int endRow,
                    int dim1){
    int idx1 = startRow;
    int i = 0, j, h = 1;
    float y;
    int rowStride = dim1;
    float *xElement, *yElement;

    //Unrolling the first few loops
    //of the transform increased speed substantially
    //(may be compiler and optimization level dependent).
    //This yields diminishing returns and does not
    //offer much improvement past 3 unrolls.
    for (idx1 = startRow; idx1 < endRow; idx1++){
        xElement = xArray + idx1 * rowStride;
        yElement = xElement + 1;
        for (i = 0; i < rowStride; i += 2){
            y = *yElement;
            *yElement = *xElement - y;
            *xElement += y;
            xElement += 2;
            yElement += 2;
        }
        if (dim1 <= 2)
            continue;
        
        xElement = xArray + idx1 * rowStride;
        yElement = xElement + 2;
	    for (i = 0; i < rowStride; i += 4){
            y = *yElement;
            *yElement = *xElement - y;
            *xElement += y;
            xElement ++;
            yElement ++;
            y = *yElement;
            *yElement = *xElement - y;
            *xElement += y;
            xElement += 3;
            yElement += 3;
        }
        if (dim1 <= 4)
            continue;

        xElement = xArray + idx1 * rowStride;
        yElement = xElement + 4;
	    for (i = 0; i < rowStride; i += 8){
            y = *yElement;
            *yElement = *xElement - y;
            *xElement += y;
            xElement ++;
            yElement ++;
            
            y = *yElement;
            *yElement = *xElement - y;
            *xElement += y;
            xElement ++;
            yElement ++;
            
            y = *yElement;
            *yElement = *xElement - y;
            *xElement += y;
            xElement ++;
            yElement ++;
            y = *yElement;
            *yElement = *xElement - y;
            *xElement += y;
            
            xElement += 5;
            yElement += 5;
        }
        if (dim1 <= 8)
            continue;

        //The general, non-unrolled transform.
        for (h = 8; h < dim1; h <<= 1){
            for (i = 0; i < rowStride; i += (h << 1)){
                xElement = xArray + idx1 * rowStride + i;
                yElement = xElement + h;
                for (j=0; j < h; j++){
                    y = *yElement;
                    *yElement = *xElement - y;
                    *xElement += y;
                    xElement++;
                    yElement++;
                }
            }
        }
    }
}



/*!
 * # floatMultiplyByDiagonalRademacherMat2D
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
void floatMultiplyByDiagonalRademacherMat2D(float *xArray,
                    int8_t *rademArray,
                    int dim1,
                    int startRow, int endRow){
    
    int i = startRow, j = i;
    float normConstant = log2(dim1) / 2;
    normConstant = 1 / pow(2, normConstant);
    int rowStride = dim1;
    float *xElement;
    int8_t *rademElement;
    
    for(i = startRow; i < endRow; i++){
	    rademElement = rademArray;
        xElement = xArray + i * rowStride;
        for (j = 0; j < rowStride; j++){
            *xElement *= *rademElement * normConstant;
            rademElement++;
            xElement++;
        }
    }
}






/*!
 * # floatMultiplyByDiagonalRademacherMat
 *
 * Multiplies an input 3d array xArray by a 3d array rademArray assumed
 * to represent a stack of diagonal matrices. rademArray should
 * therefore be of shape (a, D, C) if xArray is of shape (N, D, C).
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
 * of shape (a, D, C)
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
void floatMultiplyByDiagonalRademacherMat(float *xArray,
                    int8_t *rademArray,
                    int dim1, int dim2,
                    int startRow, int endRow){
    
    int i = startRow, j = i;
    float normConstant = log2(dim2) / 2;
    normConstant = 1 / pow(2, normConstant);
    int rowStride = dim1 * dim2;
    float *xElement;
    int8_t *rademElement;
    
    for(i = startRow; i < endRow; i++){
	    rademElement = rademArray;
        xElement = xArray + i * rowStride;
        for (j = 0; j < rowStride; j++){
            *xElement *= *rademElement * normConstant;
            rademElement++;
            xElement++;
        }
    }
}





/*!
 * # floatConv1dMultiplyByRadem
 *
 * Multiplies an input 3d array xArray by a 3d array rademArray assumed
 * to represent a stack of diagonal matrices. rademArray should
 * be of shape (a, 1, C * m) if xArray is of shape (N, D, C)
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
 * of shape (a, D, C)
 * + `reshapedDim1` The length of dim2 of xArray (e.g. D in
 * N x D x C)
 * + `reshapedDim2` The length of dim3 of xArray (e.g. C in
 * N x D x C)
 * + `startRow` The first row to modify. When multithreading,
 * the array is split into blocks such that each thread
 * modifies its own subset of the rows.
 * + `endRow` The last row to modify.
 * + `repeat` Corresponds to m from the above.
 * 
 * ## Returns:
 * Operations are in place so nothing is returned.
 */
void floatConv1dMultiplyByRadem(float *xArray,
                        int8_t *rademArray, int startRow,
                        int endRow, int reshapedDim1,
                        int reshapedDim2, int startPosition){
    int j, k;
    float normConstant = log2(reshapedDim2) / 2;
    normConstant = 1 / pow(2, normConstant);
    int rowStride = reshapedDim1 * reshapedDim2;
    float *xElement;
    int8_t *rademElement;

    for (int i = startRow; i < endRow; i++){
        xElement = xArray + i * rowStride;
        for (j = 0; j < reshapedDim1; j++){
	        rademElement = rademArray + startPosition;
            for (k = 0; k < reshapedDim2; k++){
                *xElement *= *rademElement * normConstant;
                rademElement++;
                xElement++;
            }
        }
    }
}
