/*!
 * # hadamard_transforms.cpp
 *
 * This module performs core Hadamard transform ops.
 * It includes the following functions:
 *
 * + transformRows
 * Performs the unnormalized Hadamard transform on a 3d array.
 * Can be used with 2d arrays by specifying dim1 to be 1.
 */

#include <stdint.h>
#include <math.h>
#include "hadamard_transforms.h"


/*!
 * # transformRows
 *
 * Performs an unnormalized Hadamard transform along the last
 * dimension of an input 3d array. The transform is performed
 * in place so nothing is returned. Assumes dimensions have
 * already been checked by caller. Designed to be compatible
 * with multithreading. Can be used with a 2d array by specifying
 * dim1 = 1.
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
template <typename T>
void transformRows(T __restrict xArray[], int startRow, int endRow,
                    int dim1, int dim2){
    int idx1 = startRow;
    int i = 0, j, h = 1;
    T y;
    int rowStride = dim1 * dim2;
    T *__restrict xElement, *__restrict yElement;

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
//Explicitly instantiate for external use.
template void transformRows<double>(double *__restrict xArray, int startRow, int endRow,
                    int dim1, int dim2);
template void transformRows<float>(float *__restrict xArray, int startRow, int endRow,
                    int dim1, int dim2);
