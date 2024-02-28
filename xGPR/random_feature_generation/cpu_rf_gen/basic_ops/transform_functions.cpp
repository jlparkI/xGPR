/*!
 * # transform_functions.c
 *
 * This module uses the "low-level" functions in array_operations to perform
 * fast Hadamard transforms, SORF and SRHT operations on a variety of different
 * array shapes.
 */
#include <Python.h>
#include <vector>
#include <thread>
#include "transform_functions.h"
#include "../shared_fht_functions/hadamard_transforms.h"
#include "../shared_fht_functions/shared_rfgen_ops.h"

/*!
 * # fastHadamard3dArray_
 *
 * Performs an unnormalized Hadamard transform along the last
 * dimension of an input 3d array. The transform is performed
 * in place so nothing is returned.
 *
 * ## Args:
 *
 * + `Z` Pointer to the first element of the array to be
 * modified. Must be a 3d array (N x D x C). C must be a power of 2.
 * + `zDim0` The first dimension of Z (N).
 * + `zDim1` The second dimension of Z (D)
 * + `zDim2` The third dimension of Z (C)
 * + `numThreads` The number of threads to use.
 */
template <typename T>
const char *fastHadamard3dArray_(T Z[], int zDim0, int zDim1, int zDim2,
                        int numThreads)
{
    if (numThreads > zDim0)
        numThreads = zDim0;

    std::vector<std::thread> threads(numThreads);
    int startPosition, endPosition;
    int chunkSize = (zDim0 + numThreads - 1) / numThreads;

    for (int i=0; i < numThreads; i++){
        startPosition = i * chunkSize;
        endPosition = (i + 1) * chunkSize;
        if (endPosition > zDim0)
            endPosition = zDim0;
        threads[i] = std::thread(&ThreadTransformRows3D<T>, Z, startPosition,
                                endPosition, zDim1, zDim2);
    }

    for (auto& th : threads)
        th.join();
    return "no_error";
}




/*!
 * # fastHadamard2dArray_
 *
 * Performs an unnormalized Hadamard transform along the last
 * dimension of an input 2d array. The transform is performed
 * in place so nothing is returned.
 *
 * ## Args:
 *
 * + `Z` Pointer to the first element of the array to be
 * modified. Must be a 2d array (N x C). C must be a power of 2.
 * + `zDim0` The first dimension of Z (N).
 * + `zDim1` The second dimension of Z (C)
 * + `numThreads` The number of threads to use.
 */
template <typename T>
const char *fastHadamard2dArray_(T Z[], int zDim0, int zDim1,
                        int numThreads)
{
    if (numThreads > zDim0)
        numThreads = zDim0;

    std::vector<std::thread> threads(numThreads);
    int startPosition, endPosition;
    int chunkSize = (zDim0 + numThreads - 1) / numThreads;

    for (int i=0; i < numThreads; i++){
        startPosition = i * chunkSize;
        endPosition = (i + 1) * chunkSize;
        if (endPosition > zDim0)
            endPosition = zDim0;
        threads[i] = std::thread(&ThreadTransformRows2D<T>, Z,
                        startPosition, endPosition, zDim1);
    }

    for (auto& th : threads)
        th.join();
    return "no_error";
}




/*!
 * # SORFBlockTransform_
 *
 * Performs the following operation along the last dimension
 * of the input array Z: H D1 H D2 H D3, where H is a normalized
 * Hadamard transform and D1, D2, D3 are diagonal arrays.
 *
 * ## Args:
 *
 * + `Z` Pointer to the first element of the array to be
 * modified. Must be a 3d array (N x D x C). C must be a power of 2.
 * + `radem` A stack of diagonal arrays of shape (3 x D x C).
 * + `zDim0` The first dimension of Z (N).
 * + `zDim1` The second dimension of Z (D)
 * + `zDim2` The third dimension of Z (C)
 * + `numThreads` The number of threads to use.
 */
template <typename T>
const char *SORFBlockTransform_(T Z[], int8_t *radem,
            int zDim0, int zDim1, int zDim2, int numThreads)
{
    if (numThreads > zDim0)
        numThreads = zDim0;

    std::vector<std::thread> threads(numThreads);
    int startPosition, endPosition; 
    int chunkSize = (zDim0 + numThreads - 1) / numThreads;
    
    for (int i=0; i < numThreads; i++){
        startPosition = i * chunkSize;
        endPosition = (i + 1) * chunkSize;
        if (endPosition > zDim0)
            endPosition = zDim0;
        threads[i] = std::thread(&ThreadSORFRows3D<T>, Z, radem,
                zDim1, zDim2, startPosition, endPosition);
    }

    for (auto& th : threads)
        th.join();
    return "no_error";
}







/*!
 * # SRHTBlockTransform_
 *
 * Performs the following operation along the last dimension
 * of the input array Z: H D1, where H is a normalized
 * Hadamard transform and D1 is a diagonal array.
 *
 * ## Args:
 *
 * + `Z` Pointer to the first element of the array to be
 * modified. Must be a 2d array (N x C). C must be a power of 2.
 * + `radem` A diagonal array of shape (C).
 * + `zDim0` The first dimension of Z (N).
 * + `zDim1` The second dimension of Z (C)
 * + `numThreads` The number of threads to use.
 */
template <typename T>
const char *SRHTBlockTransform_(T Z[], int8_t *radem,
            int zDim0, int zDim1, int numThreads)
{
    if (numThreads > zDim0)
        numThreads = zDim0;

    std::vector<std::thread> threads(numThreads);
    int startPosition, endPosition;
    int chunkSize = (zDim0 + numThreads - 1) / numThreads;

    for (int i=0; i < numThreads; i++){
        startPosition = i * chunkSize;
        endPosition = (i + 1) * chunkSize;
        if (endPosition > zDim0)
            endPosition = zDim0;
        threads[i] = std::thread(&ThreadSRHTRows2D<T>, Z, radem,
                zDim1, startPosition, endPosition);
    }

    for (auto& th : threads)
        th.join();
    return "no_error";
}






/*!
 * # ThreadSORFRows3D
 *
 * Performs the SORF operation for one thread for a chunk of
 * the input array Z from startRow through endRow (each thread
 * works on its own group of rows).
 *
 * ## Args:
 *
 * + `rowArgs` a void pointer to a ThreadSORFArrayArgs struct.
 * Contains all the arrays and info (e.g. startRow, endRow) needed
 * to process the chunk of the array belonging to this thread.
 */
template <typename T>
void *ThreadSORFRows3D(T arrayStart[], int8_t* rademArray,
        int dim1, int dim2, int startPosition, int endPosition){
    SORF3D(arrayStart, rademArray, startPosition, endPosition,
            dim1, dim2);
    return NULL;
}







/*!
 * # ThreadSRHTRows2D
 *
 * Performs the SRHT operation for one thread for a chunk of
 * the input array Z from startRow through endRow (each thread
 * works on its own group of rows).
 */
template <typename T>
void *ThreadSRHTRows2D(T arrayStart[], int8_t* rademArray,
        int dim1, int startPosition, int endPosition){

    multiplyByDiagonalRademacherMat2D<T>(arrayStart,
                    rademArray, dim1,
                    startPosition, endPosition);
    transformRows<T>(arrayStart, startPosition, 
                    endPosition, 1, dim1);
    return NULL;
}





/*!
 * # ThreadTransformRows3D
 *
 * Performs the fast Hadamard transform for one thread for a chunk of
 * the input array Z from startRow through endRow (each thread
 * works on its own group of rows).
 */
template <typename T>
void *ThreadTransformRows3D(T arrayStart[], int startPosition,
        int endPosition, int dim1, int dim2){
    transformRows<T>(arrayStart, startPosition, 
                    endPosition, dim1, dim2);
    return NULL;
}





/*!
 * # ThreadTransformRows2D
 *
 * Performs the fast Hadamard transform for one thread for a chunk of
 * the input array Z from startRow through endRow (each thread
 * works on its own group of rows).
 */
template <typename T>
void *ThreadTransformRows2D(T arrayStart[], int startPosition,
        int endPosition, int dim1){
    transformRows<T>(arrayStart, startPosition, 
                    endPosition, 1, dim1);
    return NULL;
}




//Instantiate the key classes that the wrapper will need to use.
template const char *fastHadamard3dArray_<double>(double Z[], int zDim0, int zDim1, int zDim2,
                        int numThreads);
template const char *fastHadamard3dArray_<float>(float Z[], int zDim0, int zDim1, int zDim2,
                        int numThreads);

template const char *fastHadamard2dArray_<float>(float Z[], int zDim0, int zDim1,
                        int numThreads);
template const char *fastHadamard2dArray_<double>(double Z[], int zDim0, int zDim1,
                        int numThreads);

template const char *SRHTBlockTransform_<float>(float Z[], int8_t *radem,
            int zDim0, int zDim1, int numThreads);
template const char *SRHTBlockTransform_<double>(double Z[], int8_t *radem,
            int zDim0, int zDim1, int numThreads);

template const char *SORFBlockTransform_<float>(float Z[], int8_t *radem, int zDim0,
        int zDim1, int zDim2, int numThreads);
template const char *SORFBlockTransform_<double>(double Z[], int8_t *radem, int zDim0,
        int zDim1, int zDim2, int numThreads);
