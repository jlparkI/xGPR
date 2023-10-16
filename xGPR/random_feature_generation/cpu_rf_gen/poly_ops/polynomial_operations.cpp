/*!
 * # polynomial_operations.cpp
 *
 * This module performs operations for polynomial kernels, including
 * exact polynomials (e.g. ExactQuadratic) and approximate.
 *
 * + cpuExactQuadratic_
 * Generates features for an exact quadratic.
 */
#include <Python.h>
#include <vector>
#include <thread>
#include "polynomial_operations.h"


#define VALID_INPUTS 0
#define INVALID_INPUTS 1
#define EPS_TOLERANCE 0.0

/*!
 * # cpuExactQuadratic_
 *
 * Generates the features for an exact quadratic. The input
 * array is not changed and all features are written to the
 * designated output array.
 *
 * ## Args:
 *
 * + `inArray` Pointer to the first element of the input array data.
 * + `inDim0` The first dimension of inArray.
 * + `inDim1` The second dimension of inArray.
 * + `numThreads` The number of threads to use.
 */
template <typename T>
const char *cpuExactQuadratic_(T inArray[], double *outArray,
                int inDim0, int inDim1, int numThreads){
    if (numThreads > inDim0)
        numThreads = inDim0;

    std::vector<std::thread> threads(numThreads);
    int startPosition, endPosition;
    int chunkSize = (inDim0 + numThreads - 1) / numThreads;

    for (int i=0; i < numThreads; i++){
        startPosition = i * chunkSize;
        endPosition = (i + 1) * chunkSize;
        if (endPosition > inDim0)
            endPosition = inDim0;
        threads[i] = std::thread(&ThreadExactQuadratic<T>, inArray, outArray,
                                startPosition, endPosition, inDim0, inDim1);
    }

    for (auto& th : threads)
        th.join();
    return "no_error";
}



/*!
 * # ThreadExactQuadratic
 *
 * Performs exact quadratic feature generation for one thread for a chunk of
 * the input array from startPosition through endPosition (each thread
 * works on its own group of rows).
 */
template <typename T>
void *ThreadExactQuadratic(T inArray[], double *outArray, int startPosition,
        int endPosition, int inDim0, int inDim1){
    exactQuadraticFeatureGen<T>(inArray, outArray,
                   startPosition, endPosition, inDim0, inDim1);
    return NULL;
}


/*!
 * # exactQuadraticFeatureGen
 *
 * Generates features for an exact quadratic, writing them
 * to the designated output array and leaving the input array
 * unchanged.
 *
 * ## Args:
 *
 * + `inArray` Pointer to the first element of the 2d input
 * data array.
 * + `outArray` Pointer to the first element of the 2d output
 * data array.
 * + `startRow` The first row to modify. When multithreading,
 * the array is split into blocks such that each thread
 * modifies its own subset of the rows.
 * + `endRow` The last row to modify.
 * + `inDim0` The length of dim0 of inArray.
 * + `inDim1` The length of dim1 of inArray.
 */
template <typename T>
void exactQuadraticFeatureGen(T inArray[], double *outArray, int startRow,
                    int endRow, int inDim0, int inDim1){
    int numInteractions = inDim1 * (inDim1 - 1) / 2;
    int outDim1 = numInteractions + 1 + 2 * inDim1;
    T *inElement = inArray + startRow * inDim1;
    double *outElement = outArray + startRow * outDim1;

    for (int i = startRow; i < endRow; i++){
        for (int j = 0; j < inDim1; j++){
            *outElement = inElement[j];
            outElement++;
            for (int k = j; k < inDim1; k++){
                *outElement = inElement[j] * inElement[k];
                outElement++;
            }
        }
        inElement += inDim1;
        outElement++;
    }
}


//Instantiate functions that the wrapper will need to use.
template const char *cpuExactQuadratic_<double>(double inArray[], double *outArray,
                int inDim0, int inDim1, int numThreads);
template const char *cpuExactQuadratic_<float>(float inArray[], double *outArray,
                int inDim0, int inDim1, int numThreads);
