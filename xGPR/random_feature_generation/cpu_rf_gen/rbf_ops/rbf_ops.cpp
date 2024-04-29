/*!
 * # rbf_ops.cpp
 *
 * This module performs all major steps involved in feature generation for
 * RBF-type kernels, which includes RBF, Matern, ARD and MiniARD (and by extension
 * the static layer kernels).
 */
#include <Python.h>
#include <stdint.h>
#include <math.h>
#include <vector>
#include <thread>
#include "rbf_ops.h"
#include "../shared_fht_functions/hadamard_transforms.h"
#include "../shared_fht_functions/shared_rfgen_ops.h"



/*!
 * # rbfFeatureGen_
 *
 * Generates features for the input array and stores them in outputArray.
 *
 * ## Args:
 *
 * + `cArray` Pointer to the first element of the input array.
 * Must be a 3d array (N x D x C). C must be a power of 2.
 * + `radem` Pointer to first element of int8_t stack of diagonal.
 * arrays. Must be shape (3 x D x C).
 * + `chiArr` Pointer to first element of diagonal array to ensure
 * correct marginals. Must be of shape numFreqs.
 * + `outputArray` Pointer to first element of output array. Must
 * be a 2d array (N x 2 * numFreqs).
 * + `rbfNormConstant` A value by which all outputs are multipled.
 * Should be beta hparam * sqrt(1 / numFreqs). Is calculated by
 * caller.
 * + `dim0` shape[0] of input array
 * + `dim1` shape[1] of input array
 * + `dim2` shape[2] of input array
 * + `numFreqs` (numRFFs / 2) -- the number of frequencies to sample.
 * + `numThreads` The number of threads to use.
 */
template <typename T>
const char *rbfFeatureGen_(T cArray[], int8_t *radem,
                T chiArr[], double *outputArray,
                double rbfNormConstant,
                int dim0, int dim1, int rademShape2,
                int numFreqs, int numThreads,
                int paddedBufferSize){
    if (numThreads > dim0)
        numThreads = dim0;

    std::vector<std::thread> threads(numThreads);
    int startPosition, endPosition;
    
    int chunkSize = (dim0 + numThreads - 1) / numThreads;

    for (int i=0; i < numThreads; i++){
        startPosition = i * chunkSize;
        endPosition = (i + 1) * chunkSize;
        if (endPosition > dim0)
            endPosition = dim0;
        
        threads[i] = std::thread(&allInOneRBFGen<T>, cArray,
                radem, chiArr, outputArray, dim1, numFreqs,
                rademShape2, startPosition, endPosition,
                paddedBufferSize, rbfNormConstant);
    }

    for (auto& th : threads)
        th.join();
    return "no_error";
}
//Explicitly instantiate so wrapper can use.
template const char *rbfFeatureGen_<double>(double cArray[], int8_t *radem,
                double chiArr[], double *outputArray,
                double rbfNormConstant,
                int dim0, int dim1, int rademShape2,
                int numFreqs, int numThreads,
                int paddedBufferSize);
template const char *rbfFeatureGen_<float>(float cArray[], int8_t *radem,
                float chiArr[], double *outputArray,
                double rbfNormConstant,
                int dim0, int dim1, int rademShape2,
                int numFreqs, int numThreads,
                int paddedBufferSize);


/*!
 * # rbfGrad_
 *
 * Generates features for the input array and stores them in outputArray.
 *
 * ## Args:
 *
 * + `cArray` Pointer to the first element of the input array.
 * Must be a 3d array (N x D x C). C must be a power of 2.
 * + `radem` Pointer to first element of int8_t stack of diagonal.
 * arrays. Must be shape (3 x D x C).
 * + `chiArr` Pointer to first element of diagonal array to ensure
 * correct marginals. Must be of shape numFreqs.
 * + `outputArray` Pointer to first element of output array. Must
 * be a 2d array (N x 2 * numFreqs).
 * + `gradientArray` Pointer to the first element of gradient array.
 * Must be a 2d array (N x 2 * numFreqs).
 * + `rbfNormConstant` A value by which all outputs are multipled.
 * Should be beta hparam * sqrt(1 / numFreqs). Is calculated by
 * caller.
 * + `sigma` The lengthscale hyperparameter.
 * + `dim0` shape[0] of input array
 * + `dim1` shape[1] of input array
 * + `rademShape2` shape[2] of radem
 * + `numFreqs` (numRFFs / 2) -- the number of frequencies to sample.
 * + `numThreads` The number of threads to use.
 */
template <typename T>
const char *rbfGrad_(T cArray[], int8_t *radem,
                T chiArr[], double *outputArray,
                double *gradientArray,
                double rbfNormConstant, T sigma,
                int dim0, int dim1, int rademShape2,
                int numFreqs, int numThreads,
                int paddedBufferSize){
    if (numThreads > dim0)
        numThreads = dim0;

    std::vector<std::thread> threads(numThreads);
    int startPosition, endPosition;
    int chunkSize = (dim0 + numThreads - 1) / numThreads;

    for (int i=0; i < numThreads; i++){
        startPosition = i * chunkSize;
        endPosition = (i + 1) * chunkSize;
        if (endPosition > dim0)
            endPosition = dim0;
 
        threads[i] = std::thread(&allInOneRBFGrad<T>, cArray,
                radem, chiArr, outputArray,
                gradientArray, dim1, numFreqs,
                rademShape2, startPosition,
                endPosition, paddedBufferSize,
                rbfNormConstant, sigma);
    }

    for (auto& th : threads)
        th.join();
    return "no_error";
}
//Explicitly instantiate for external use.
template const char *rbfGrad_<double>(double cArray[], int8_t *radem,
                double chiArr[], double *outputArray,
                double *gradientArray,
                double rbfNormConstant, double sigma,
                int dim0, int dim1, int rademShape2,
                int numFreqs, int numThreads,
                int paddedBufferSize);
template const char *rbfGrad_<float>(float cArray[], int8_t *radem,
                float chiArr[], double *outputArray,
                double *gradientArray,
                double rbfNormConstant, float sigma,
                int dim0, int dim1, int rademShape2,
                int numFreqs, int numThreads,
                int paddedBufferSize);




/*!
 * # allInOneRBFGen
 *
 * Performs the RBF-based kernel feature generation
 * process for the input, for one thread.
 */
template <typename T>
void *allInOneRBFGen(T xdata[], int8_t *rademArray, T chiArr[],
        double *outputArray, int dim1, int numFreqs, int rademShape2,
        int startRow, int endRow, int paddedBufferSize,
        double scalingTerm){

    int numRepeats = (numFreqs + paddedBufferSize - 1) / paddedBufferSize;
    //Notice that we don't have error handling here...very naughty. Out of
    //memory should be extremely rare since we are only allocating memory
    //for one row of the convolution. TODO: add error handling here.
    T *copyBuffer = new T[paddedBufferSize];
    T *xElement;

    for (int i=startRow; i < endRow; i++){

        int repeatPosition = 0;
        xElement = xdata + i * dim1;

        for (int k=0; k < numRepeats; k++){
            for (int m=0; m < dim1; m++)
                copyBuffer[m] = xElement[m];
            for (int m=dim1; m < paddedBufferSize; m++)
                copyBuffer[m] = 0;

            singleVectorSORF(copyBuffer, rademArray, repeatPosition,
                        rademShape2, paddedBufferSize);
            singleVectorRBFPostProcess(copyBuffer, chiArr, outputArray,
                        paddedBufferSize, numFreqs, i, k, scalingTerm);
            repeatPosition += paddedBufferSize;
        }
    }
    delete[] copyBuffer;

    return NULL;
}


/*!
 * # allInOneRBFGrad
 *
 * Performs the RBF-based kernel feature generation
 * process for the input, for one thread, and calculates the
 * gradient, which is stored in a separate array.
 */
template <typename T>
void *allInOneRBFGrad(T xdata[], int8_t *rademArray, T chiArr[],
        double *outputArray, double *gradientArray,
        int dim1, int numFreqs, int rademShape2, int startRow,
        int endRow, int paddedBufferSize,
        double scalingTerm, T sigma){

    int numRepeats = (numFreqs + paddedBufferSize - 1) / paddedBufferSize;
    //Notice that we don't have error handling here...very naughty. Out of
    //memory should be extremely rare since we are only allocating memory
    //for one row of the convolution. TODO: add error handling here.
    T *copyBuffer = new T[paddedBufferSize];
    T *xElement;

    for (int i=startRow; i < endRow; i++){
        int repeatPosition = 0;
        xElement = xdata + i * dim1;

        for (int k=0; k < numRepeats; k++){
            for (int m=0; m < dim1; m++)
                copyBuffer[m] = xElement[m];
            for (int m=dim1; m < paddedBufferSize; m++)
                copyBuffer[m] = 0;

            singleVectorSORF(copyBuffer, rademArray, repeatPosition,
                        rademShape2, paddedBufferSize);
            singleVectorRBFPostGrad(copyBuffer, chiArr, outputArray,
                        gradientArray, sigma, paddedBufferSize, numFreqs,
                        i, k, scalingTerm);
            repeatPosition += paddedBufferSize;
        }
    }
    delete[] copyBuffer;

    return NULL;
}
