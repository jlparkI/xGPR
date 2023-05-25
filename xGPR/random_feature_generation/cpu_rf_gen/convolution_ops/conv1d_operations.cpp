/*!
 * # conv1d_operations.c
 *
 * This module performs operations unique to the convolution
 * kernels in xGPR, essentially orthogonal random features based
 * convolution. It includes the following functions:
 *
 * + conv1dPrep_
 * Performs the core fast hadamard transform based operations needed
 * for convolution with SORF.
 *
 * + FloatThreadConv1d
 * Called once by conv1dPrep_ for each thread.
 */
#include <Python.h>
#include <vector>
#include <thread>
#include <math.h>
#include "conv1d_operations.h"
#include "../shared_fht_functions/basic_array_operations.h"





/*!
 * # conv1dPrep_
 *
 * Performs key steps for orthogonal random features based convolution
 * with multithreading, when input is a single-precision array.
 * It is assumed that caller has checked dimensions and they are all correct.
 *
 * ## Args:
 *
 * + `radem` The stacks of diagonal matrices used in
 * the transform. Must be of shape (3 x 1 x m * C) where m is
 * an integer that indicates the number of times we must repeat
 * the operation to generate the requested number of sampled frequencies.
 * + `reshapedX` Pointer to the first element of the array that will
 * be used for the convolution. Shape is (N x D x C). C must be
 * a power of 2.
 * + `numThreads` The number of threads to use
 * + `reshapedDim0` The first dimension of reshapedX
 * + `reshapedDim1` The second dimension of reshapedX
 * + `reshapedDim2` The last dimension of reshapedX
 * + `startPosition` Where to start when reading through radem.
 * + `numFreqs` The number of frequencies to sample.
 * numFreqs must be equal to shape[2] of radem.
 *
 * ## Returns:
 * "error" if an error, "no_error" otherwise.
 */
template <typename T>
const char *conv1dPrep_(int8_t *radem, T reshapedX[],
            int numThreads, int reshapedDim0,
            int reshapedDim1, int reshapedDim2,
            int startPosition, int numFreqs)
{

    if (numThreads > reshapedDim0)
        numThreads = reshapedDim0;

    std::vector<std::thread> threads(numThreads);
    int startRow, endRow;
    int chunkSize = (reshapedDim0 + numThreads - 1) / numThreads;
    
    for (int i=0; i < numThreads; i++){
        startPosition = startPosition;
        startRow = i * chunkSize;
        endRow = (i + 1) * chunkSize;
        if (endRow > reshapedDim0)
            endRow = reshapedDim0;
        threads[i] = std::thread(&threadConv1d<T>, reshapedX, radem,
                reshapedDim1, reshapedDim2, numFreqs, startRow,
                endRow, startPosition);
    }

    for (auto& th : threads)
        th.join();
    return "no_error";

}






/*!
 * # threadConv1d
 *
 * Performs orthogonal random features based convolution
 * for a single thread. reshapedX is split up into
 * num_threads chunks each with a start and end row.
 *
 * ## Args:
 * + `sharedArgs` A void pointer to a struct
 * containing pointers to the arrays needed to execute the
 * transform, the start and end rows etc.
 */
template <typename T>
void *threadConv1d(T reshapedXArray[], int8_t* rademArray,
        int reshapedDim1, int reshapedDim2, int numFreqs,
        int startRow, int endRow, int startPosition){
    
    conv1dMultiplyByRadem<T>(reshapedXArray,
                    rademArray, startRow,
                    endRow, reshapedDim1, 
                    reshapedDim2, startPosition);
    transformRows3D<T>(reshapedXArray, startRow,
                    endRow, reshapedDim1, 
                    reshapedDim2);
    conv1dMultiplyByRadem<T>(reshapedXArray,
                    rademArray + numFreqs,
                    startRow, endRow,
                    reshapedDim1, reshapedDim2,
                    startPosition);
    transformRows3D<T>(reshapedXArray, startRow,
                    endRow, reshapedDim1, 
                    reshapedDim2);
    
    conv1dMultiplyByRadem<T>(reshapedXArray,
                    rademArray + 2 * numFreqs,
                    startRow, endRow,
                    reshapedDim1, reshapedDim2,
                    startPosition);
    transformRows3D<T>(reshapedXArray, startRow,
                    endRow, reshapedDim1, 
                    reshapedDim2);
    return NULL;
}

//Instantiate the functions the wrapper will need to access.
template const char *conv1dPrep_<float>(int8_t *radem, float reshapedX[],
            int numThreads, int reshapedDim0,
            int reshapedDim1, int reshapedDim2,
            int startPosition, int numFreqs);
template const char *conv1dPrep_<double>(int8_t *radem, double reshapedX[],
            int numThreads, int reshapedDim0,
            int reshapedDim1, int reshapedDim2,
            int startPosition, int numFreqs);
