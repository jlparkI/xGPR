/*!
 * # hadamard_transforms.cpp
 *
 * This module performs core Hadamard transform ops.
 */
#include <stdint.h>
#include <math.h>
#include "hadamard_transforms.h"

template <typename T>
void smallBlockTransform(T xArray[], int startRow, int endRow,
                    int dim1, int dim2){
    T y;
    int rowStride = dim1 * dim2;
    T *__restrict xElement;

    for (int idx1 = startRow; idx1 < endRow; idx1++){
        xElement = xArray + idx1 * rowStride;
        for (int i = 0; i < rowStride; i += 2){
            y = xElement[1];
            xElement[1] = *xElement - y;
            *xElement += y;
            xElement += 2;
        }
        if (dim2 <= 2)
            continue;
        
        xElement = xArray + idx1 * rowStride;
	    for (int i = 0; i < rowStride; i += 4){
            y = xElement[2];
            xElement[2] = *xElement - y;
            *xElement += y;
            xElement ++;
            y = xElement[2];
            xElement[2] = *xElement - y;
            *xElement += y;
            xElement += 3;
        }
        if (dim2 <= 4)
            continue;

        xElement = xArray + idx1 * rowStride;
	    for (int i = 0; i < rowStride; i += 8){
            y = xElement[4];
            xElement[4] = *xElement - y;
            *xElement += y;
            xElement ++;
            
            y = xElement[4];
            xElement[4] = *xElement - y;
            *xElement += y;
            xElement ++;
            
            y = xElement[4];
            xElement[4] = *xElement - y;
            *xElement += y;
            xElement ++;
            y = xElement[4];
            xElement[4] = *xElement - y;
            *xElement += y;
            
            xElement += 5;
        }
    }
}
template void smallBlockTransform<double>(double *xArray, int startRow, int endRow,
                    int dim1, int dim2);
template void smallBlockTransform<float>(float *xArray, int startRow, int endRow,
                    int dim1, int dim2);


template <typename T>
void generalTransform(T xArray[], int startRow, int endRow,
                    int dim1, int dim2){
    T y;
    T yGrp[8];
    int rowStride = dim1 * dim2;
    T *__restrict xElement;

    for (int idx1 = startRow; idx1 < endRow; idx1++){
        for (int idx2 = 0; idx2 < dim1; idx2++){

            //We unroll the first few loops to make this very easy
            //for compiler to optimize; this provides a small speedup.
            xElement = xArray + idx1 * rowStride + idx2 * dim2;
            for (int i=0; i < dim2; i += 8){

                #pragma omp simd
                for (int k=0; k < 8; k+=2){
                    yGrp[k] = xElement[k] + xElement[k+1];
                    yGrp[k+1] = xElement[k] - xElement[k+1];
                }
                
                y = yGrp[2];
                yGrp[2] = yGrp[0] - y;
                yGrp[0] += y;
                y = yGrp[3];
                yGrp[3] = yGrp[1] - y;
                yGrp[1] += y;

                y = yGrp[6];
                yGrp[6] = yGrp[4] - y;
                yGrp[4] += y;
                y = yGrp[7];
                yGrp[7] = yGrp[5] - y;
                yGrp[5] += y;

                #pragma omp simd
                for (int k=0; k < 4; k++)
                    xElement[k] = yGrp[k] + yGrp[k+4];

                #pragma omp simd
                for (int k=4; k < 8; k++)
                    xElement[k] = yGrp[k-4] - yGrp[k];

                xElement += 8;
            }

            //The general, non-unrolled transform.
            for (int h = 8; h < dim2; h <<= 1){
                for (int i = 0; i < dim2; i += (h << 1)){
                    xElement = xArray + idx1 * rowStride + idx2 * dim2 + i;
                    for (int j=0; j < h; j++){
                        y = xElement[h];
                        xElement[h] = *xElement - y;
                        *xElement += y;
                        xElement++;
                    }
                }
            }
        }
    }
}


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
    switch (dim2){
        case 2:
        case 4:
        case 8:
            smallBlockTransform(xArray, startRow, endRow, dim1, dim2);
            break;
        default:
            generalTransform(xArray, startRow, endRow, dim1, dim2);
            break;
    }
}
template void transformRows<double>(double *__restrict xArray, int startRow, int endRow,
                    int dim1, int dim2);
template void transformRows<float>(float *__restrict xArray, int startRow, int endRow,
                    int dim1, int dim2);






/*!
 * # singleVectorTransform
 *
 * Performs an unnormalized Hadamard transform along a single
 * vector, which allows for some simplifications.
 *
 * ## Args:
 *
 * + `xArray` Pointer to the first element of the array to be
 * modified. Must be a 1d array (e.g. C). C MUST be
 * a power of 2.
 * + `dim` The length of the array.
 */
template <typename T>
void singleVectorTransform(T xArray[], int dim){
    T y;
    T *__restrict xElement;
    
    xElement = xArray;
    for (int i = 0; i < dim; i += 2){
        y = xElement[1];
        xElement[1] = *xElement - y;
        *xElement += y;
        xElement += 2;
    }
    if (dim <= 2){
        return;
    }
        
    xElement = xArray;
	for (int i = 0; i < dim; i += 4){
        y = xElement[2];
        xElement[2] = *xElement - y;
        *xElement += y;
        xElement ++;
        y = xElement[2];
        xElement[2] = *xElement - y;
        *xElement += y;
        xElement += 3;
    }    
    if (dim <= 4){
        return;
    }

    xElement = xArray;
	for (int i = 0; i < dim; i += 8){
        y = xElement[4];
        xElement[4] = *xElement - y;
        *xElement += y;
        xElement ++;
            
        y = xElement[4];
        xElement[4] = *xElement - y;
        *xElement += y;
        xElement ++;
            
        y = xElement[4];
        xElement[4] = *xElement - y;
        *xElement += y;
        xElement ++;
        y = xElement[4];
        xElement[4] = *xElement - y;
        *xElement += y;
            
        xElement += 5;
    }
    if (dim <= 8){
        return;
    }


    //The general, non-unrolled transform.
    for (int h = 8; h < dim; h <<= 1){
        for (int i = 0; i < dim; i += (h << 1)){
            xElement = xArray + i;
            for (int j=0; j < h; j++){
                y = xElement[h];
                xElement[h] = *xElement - y;
                *xElement += y;
                xElement++;
            }
        }
    }
}
template void singleVectorTransform<double>(double *__restrict xArray, int dim);
template void singleVectorTransform<float>(float *__restrict xArray, int dim);
