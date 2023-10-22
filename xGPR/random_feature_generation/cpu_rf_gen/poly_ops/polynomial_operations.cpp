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
#include "../shared_fht_functions/hadamard_transforms.h"
#include "../shared_fht_functions/diagonal_matmul_ops.h"


#define VALID_INPUTS 0
#define INVALID_INPUTS 1



/*!
 * # approxPolynomial_
 * Performs all steps required to generate random features for classic
 * approximate polynomial kernels.
 *
 * ## Args:
 *
 * + `radem` The stacks of diagonal matrices used in
 * the transform. Must be of shape (3 * m x D x C) where m is
 * the degree of the polynomial.
 * + `reshapedX` Pointer to the first element of the array that will
 * be used as input data. A copy of this array is modified
 * rather than the original. Shape is (N x D x C). C must be
 * a power of 2.
 * + `copyBuffer` An array of the same size and shape as reshapedX into
 * which reshapedX can be copied. copyBuffer can then be modified in place
 * to generate the random features.
 * + `chiArr` A diagonal array that will be multiplied against the output
 * of the SORF operation. Of shape (m, D, C).
 * + `outputArray` The output array. Modified in place.
 * + `numThreads` The number of threads to use
 * + `polydegree` The degree of the polynomial.
 * + `reshapedDim0` The first dimension of reshapedX
 * + `reshapedDim1` The second dimension of reshapedX
 * + `reshapedDim2` The last dimension of reshapedX
 * + `numFreqs` The number of frequencies to sample. Must be <=
 * radem.shape[2].
 *
 * ## Returns:
 * An error message if an error is encountered, "no_error" otherwise.
 */
template <typename T>
const char *approxPolynomial_(int8_t *radem, T reshapedX[],
            T copyBuffer[], T chiArr[], double *outputArray,
            int numThreads, int polydegree, int reshapedDim0,
            int reshapedDim1, int reshapedDim2,
            int numFreqs)
{
    if (numThreads > reshapedDim0)
        numThreads = reshapedDim0;

    std::vector<std::thread> threads(numThreads);
    int startRow, endRow;
    int chunkSize = (reshapedDim0 + numThreads - 1) / numThreads;
    
    for (int i=0; i < numThreads; i++){
        startRow = i * chunkSize;
        endRow = (i + 1) * chunkSize;
        if (endRow > reshapedDim0)
            endRow = reshapedDim0;
        threads[i] = std::thread(&threadApproxPolynomial<T>, reshapedX,
                copyBuffer, radem, chiArr, outputArray,
                polydegree, reshapedDim1, reshapedDim2, numFreqs,
                startRow, endRow);
    }

    for (auto& th : threads)
        th.join();
    return "no_error";
}
//Instantiate functions that the wrapper will need to use.
template const char *approxPolynomial_<double>(int8_t *radem, double reshapedX[],
            double copyBuffer[], double chiArr[], double *outputArray,
            int numThreads, int polydegree, int reshapedDim0,
            int reshapedDim1, int reshapedDim2,
            int numFreqs);
template const char *approxPolynomial_<float>(int8_t *radem, float reshapedX[],
            float copyBuffer[], float chiArr[], double *outputArray,
            int numThreads, int polydegree, int reshapedDim0,
            int reshapedDim1, int reshapedDim2,
            int numFreqs);




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
    int startRow, endRow;
    int chunkSize = (inDim0 + numThreads - 1) / numThreads;

    for (int i=0; i < numThreads; i++){
        startRow = i * chunkSize;
        endRow = (i + 1) * chunkSize;
        if (endRow > inDim0)
            endRow = inDim0;
        threads[i] = std::thread(&ThreadExactQuadratic<T>, inArray, outArray,
                                startRow, endRow, inDim0, inDim1);
    }

    for (auto& th : threads)
        th.join();
    return "no_error";
}
//Instantiate functions that the wrapper will need to use.
template const char *cpuExactQuadratic_<double>(double inArray[], double *outArray,
                int inDim0, int inDim1, int numThreads);
template const char *cpuExactQuadratic_<float>(float inArray[], double *outArray,
                int inDim0, int inDim1, int numThreads);


/*!
 * # threadApproxPolynomial
 *
 * Performs approximate polynomial feature generation for one thread for a chunk
 * of the input array from startRow through endRow (each thread works
 * on its own group of rows).
 */
template <typename T>
void *threadApproxPolynomial(T inArray[], T copyBuffer[], int8_t *radem,
        T chiArr[], double *outputArray, int polydegree, int dim1,
        int dim2, int numFreqs, int startRow, int endRow){
    int rowSize = dim1 * dim2;

    // Copy the input array to the copy buffer and perform the SORF
    // operation on it.
    multiplyByDiagonalRademAndCopy(inArray, copyBuffer,
                    radem, dim1, dim2,
                    startRow, endRow);
    transformRows3D<T>(copyBuffer, startRow, 
                    endRow, dim1, dim2);
    multiplyByDiagonalRademacherMat<T>(copyBuffer,
                    radem + rowSize, dim1, dim2, 
                    startRow, endRow);
    transformRows3D<T>(copyBuffer, startRow, 
                    endRow, dim1, dim2);
    multiplyByDiagonalRademacherMat<T>(copyBuffer,
                    radem + 2 * rowSize, dim1, dim2, 
                    startRow, endRow);
    transformRows3D<T>(copyBuffer, startRow, 
                    endRow, dim1, dim2);
    // Now transfer it to the output array.
    outArrayCopyTransfer(copyBuffer, outputArray, chiArr, dim1,
            dim2, numFreqs, startRow, endRow);


    // Now repeat this process for up to polydegree times,
    // using the appropriate rows of chiArr and radem.
    for (int i = 1; i < polydegree; i++){
        multiplyByDiagonalRademAndCopy(inArray, copyBuffer,
                    radem + (3 * i) * rowSize, dim1, dim2,
                    startRow, endRow);
        transformRows3D<T>(copyBuffer, startRow, 
                    endRow, dim1, dim2);
        multiplyByDiagonalRademacherMat<T>(copyBuffer,
                    radem + (3 * i + 1) * rowSize, dim1, dim2, 
                    startRow, endRow);
        transformRows3D<T>(copyBuffer, startRow, 
                    endRow, dim1, dim2);
        multiplyByDiagonalRademacherMat<T>(copyBuffer,
                    radem + (3 * i + 2) * rowSize, dim1, dim2, 
                    startRow, endRow);
        transformRows3D<T>(copyBuffer, startRow, 
                    endRow, dim1, dim2);
        outArrayMatTransfer(copyBuffer, outputArray, chiArr, dim1,
                dim2, numFreqs, startRow, endRow, i);
    }
    return NULL;
}



/*!
 * # ThreadExactQuadratic
 *
 * Performs exact quadratic feature generation for one thread for a chunk of
 * the input array from startRow through endRow (each thread
 * works on its own group of rows).
 */
template <typename T>
void *ThreadExactQuadratic(T inArray[], double *outArray, int startRow,
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
    return NULL;
}


/*!
 * # outArrayMatTransfer
 *
 * Multiplies the contents of the output array by copyBuffer while
 * simultaneously multiplying by chiArr and cutting off any excess
 * (elements in excess of those needed given the requested number
 * of RFFs). The appropriate row of chiArr is used.
 */
template <typename T>
void *outArrayMatTransfer(T copyBuffer[], double *outArray, T chiArr[],
        int dim1, int dim2, int numFreqs, int startRow, int endRow,
        int chiArrRow){
    T *inElement, *chiArrElement;
    double *outElement;

    for (int i = startRow; i < endRow; i++){
        chiArrElement = chiArr + chiArrRow * dim1 * dim2;
        outElement = outArray + i * numFreqs;
        inElement = copyBuffer + i * dim1 * dim2;
        for (int j = 0; j < numFreqs; j++){
            *outElement *= *inElement * *chiArrElement;
            outElement++;
            inElement++;
            chiArrElement++;
        }
    }
    return NULL;
}



/*!
 * # outArrayCopyTransfer
 *
 * Copies the contents of copyBuffer into the output array while
 * simultaneously multiplying by chiArr and cutting off any excess
 * (elements in excess of those needed given the requested number
 * of RFFs). The first row of chiArr is used.
 */
template <typename T>
void *outArrayCopyTransfer(T copyBuffer[], double *outArray, T chiArr[],
        int dim1, int dim2, int numFreqs, int startRow, int endRow){
    T *inElement, *chiArrElement;
    double *outElement;

    for (int i = startRow; i < endRow; i++){
        chiArrElement = chiArr;
        outElement = outArray + i * numFreqs;
        inElement = copyBuffer + i * dim1 * dim2;
        for (int j = 0; j < numFreqs; j++){
            *outElement = *inElement * *chiArrElement;
            outElement++;
            inElement++;
            chiArrElement++;
        }
    }
    return NULL;
}
