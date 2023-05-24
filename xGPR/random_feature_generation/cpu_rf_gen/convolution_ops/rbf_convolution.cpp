/*!
 * # rbf_convolution.c
 *
 * This module performs operations unique to the RBF-based convolution
 * kernels in xGPR, which includes FHTConv1d and GraphConv1d. It
 * contains the following functions:
 *
 * + doubleConvRBFFeatureGen_
 * Performs all steps required to generate random features for RBF-based
 * convolution kernels (FHTConv1d, GraphConv1d, ARD variants of GraphConv1d).
 *
 * + floatConvRBFFeatureGen_
 * Performs all steps required to generate random features for RBF-based
 * convolution kernels (FHTConv1d, GraphConv1d, ARD variants of GraphConv1d).
 *
 * + doubleConvRBFGrad_
 * Performs all steps required to generate random features for RBF-based
 * convolution kernels (FHTConv1d, GraphConv1d, ARD variants of GraphConv1d)
 * and additionally calculate the gradient w/r/t lengthscale.
 *
 * + floatConvRBFGrad_
 * Performs all steps required to generate random features for RBF-based
 * convolution kernels (FHTConv1d, GraphConv1d, ARD variants of GraphConv1d)
 * and additionally calculate the gradient w/r/t lengthscale.
 * 
 * + FloatThreadConvRBFGen
 * Called once for each thread when generating RBF-based convolution
 * features.
 *
 * + DoubleThreadConvRBFGen
 * Called once for each thread when generating RBF-based convolution
 * features.
 *
 * + FloatThreadConvRBFGrad
 * Called once for each thread when generating RBF-based convolution
 * features with additional gradient calculation.
 *
 * + DoubleThreadConvRBFGrad
 * Called once for each thread when generating RBF-based convolution
 * features with additional gradient calculation.
 * 
 * + RBFGenPostProcess
 * Performs the last step in RBF-based convolution feature generation.
 *
 * + RBFGenGrad
 * Performs the last step in RBF-based convolution feature generation
 * and gradient calculation.
 * 
 * Functions from float_ and double_ array_operations.c are called to
 * perform the Hadamard transform and diagonal matrix multiplications.
 */
#include <Python.h>
#include <thread>
#include <vector>
#include <math.h>
#include "rbf_convolution.h"
#include "../shared_fht_functions/basic_array_operations.h"



/*!
 * # doubleConvRBFFeatureGen_
 * Performs all steps required to generate random features for RBF-based
 * convolution kernels (FHTConv1d, GraphConv1d, ARD variants of GraphConv1d).
 * It is assumed that caller has checked dimensions and they are all correct.
 *
 * ## Args:
 *
 * + `radem` The stacks of diagonal matrices used in
 * the transform. Must be of shape (3 x 1 x m * C) where m is
 * an integer that indicates the number of times we must repeat
 * the operation to generate the requested number of sampled frequencies.
 * + `reshapedX` Pointer to the first element of the array that will
 * be used for the convolution. A copy of this array is modified
 * rather than the original. Shape is (N x D x C). C must be
 * a power of 2.
 * + `copyBuffer` An array of the same size and shape as reshapedX into
 * which reshapedX can be copied. copyBuffer can then be modified in place
 * to generate the random features.
 * + `chiArr` A diagonal array that will be multiplied against the output
 * of the SORF operation. Must be of shape numFreqs.
 * + `outputArray` The output array. Must be of shape (N, 2 * numFreqs).
 * + `numThreads` The number of threads to use
 * + `reshapedDim0` The first dimension of reshapedX
 * + `reshapedDim1` The second dimension of reshapedX
 * + `reshapedDim2` The last dimension of reshapedX
 * + `numFreqs` The number of frequencies to sample. Must be <=
 * radem.shape[2].
 * + `rademShape2` The number of elements in one row of radem.
 *
 * ## Returns:
 * "error" if an error, "no_error" otherwise.
 */
const char *doubleConvRBFFeatureGen_(int8_t *radem, double *reshapedX,
            double *copyBuffer, double *chiArr, double *outputArray,
            int numThreads, int reshapedDim0,
            int reshapedDim1, int reshapedDim2,
            int numFreqs, int rademShape2)
{
    if (numThreads > reshapedDim0)
        numThreads = reshapedDim0;

    struct ThreadConvRBFDoubleArgs *th_args = (ThreadConvRBFDoubleArgs*)malloc(numThreads * sizeof(struct ThreadConvRBFDoubleArgs));
    if (th_args == NULL){
        PyErr_SetString(PyExc_ValueError, "System out of memory! RUN FOR YOUR LIFE!!!!");
        return "error";
    }
    std::vector<std::thread> threads(numThreads);
    int chunkSize = (reshapedDim0 + numThreads - 1) / numThreads;
    
    //We assume here that numFreqs is an integer multiple of reshapedDim2.
    //Caller must check this -- the Cython wrapper does.
    for (int i=0; i < numThreads; i++){
        th_args[i].startRow = i * chunkSize;
        th_args[i].endRow = (i + 1) * chunkSize;
        if (th_args[i].endRow > reshapedDim0)
            th_args[i].endRow = reshapedDim0;
        th_args[i].reshapedDim1 = reshapedDim1;
        th_args[i].reshapedDim2 = reshapedDim2;
        th_args[i].numFreqs = numFreqs;
        th_args[i].rademArray = radem;
        th_args[i].reshapedXArray = reshapedX;
        th_args[i].chiArr = chiArr;
        th_args[i].copyBuffer = copyBuffer;
        th_args[i].outputArray = outputArray;
        th_args[i].rademShape2 = rademShape2;
    }

    for (int i=0; i < numThreads; i++){
        threads[i] = std::thread(doubleThreadConvRBFGen, &th_args[i]);
    }

    for (auto& th : threads)
        th.join();
    free(th_args);
    return "no_error";
}



/*!
 * # floatConvRBFFeatureGen_
 * Performs all steps required to generate random features for RBF-based
 * convolution kernels (FHTConv1d, GraphConv1d, ARD variants of GraphConv1d).
 * It is assumed that caller has checked dimensions and they are all correct.
 *
 * ## Args:
 *
 * + `radem` The stacks of diagonal matrices used in
 * the transform. Must be of shape (3 x 1 x m * C) where m is
 * an integer that indicates the number of times we must repeat
 * the operation to generate the requested number of sampled frequencies.
 * + `reshapedX` Pointer to the first element of the array that will
 * be used for the convolution. A copy of this array is modified
 * rather than the original. Shape is (N x D x C). C must be
 * a power of 2.
 * + `copyBuffer` An array of the same size and shape as reshapedX into
 * which reshapedX can be copied. copyBuffer can then be modified in place
 * to generate the random features.
 * + `chiArr` A diagonal array that will be multiplied against the output
 * of the SORF operation. Of shape numFreqs.
 * + `outputArray` The output array. Of shape (N, 2 * numFreqs).
 * + `numThreads` The number of threads to use
 * + `reshapedDim0` The first dimension of reshapedX
 * + `reshapedDim1` The second dimension of reshapedX
 * + `reshapedDim2` The last dimension of reshapedX
 * + `numFreqs` The number of frequencies to sample. Must be <=
 * radem.shape[2].
 * + `rademShape2` The number of elements in one row of radem.
 *
 * ## Returns:
 * "error" if an error, "no_error" otherwise.
 */
const char *floatConvRBFFeatureGen_(int8_t *radem, float *reshapedX,
            float *copyBuffer, float *chiArr, double *outputArray,
            int numThreads, int reshapedDim0,
            int reshapedDim1, int reshapedDim2,
            int numFreqs, int rademShape2)
{
    if (numThreads > reshapedDim0)
        numThreads = reshapedDim0;

    struct ThreadConvRBFFloatArgs *th_args = (ThreadConvRBFFloatArgs*)malloc(numThreads * sizeof(struct ThreadConvRBFFloatArgs));
    if (th_args == NULL){
        PyErr_SetString(PyExc_ValueError, "System out of memory!");
        return "error";
    }
    std::vector<std::thread> threads(numThreads);
    int chunkSize = (reshapedDim0 + numThreads - 1) / numThreads;
    
    //We assume here that numFreqs is an integer multiple of reshapedDim2.
    //Caller must check this -- the Cython wrapper does.
    for (int i=0; i < numThreads; i++){
        th_args[i].startRow = i * chunkSize;
        th_args[i].endRow = (i + 1) * chunkSize;
        if (th_args[i].endRow > reshapedDim0)
            th_args[i].endRow = reshapedDim0;
        th_args[i].reshapedDim1 = reshapedDim1;
        th_args[i].reshapedDim2 = reshapedDim2;
        th_args[i].numFreqs = numFreqs;
        th_args[i].rademArray = radem;
        th_args[i].reshapedXArray = reshapedX;
        th_args[i].chiArr = chiArr;
        th_args[i].copyBuffer = copyBuffer;
        th_args[i].outputArray = outputArray;
        th_args[i].rademShape2 = rademShape2;
    }

    for (int i=0; i < numThreads; i++){
        threads[i] = std::thread(floatThreadConvRBFGen, &th_args[i]);
    }

    for (auto& th : threads)
        th.join();
    free(th_args);
    return "no_error";
}


/*!
 * # doubleConvRBFGrad_
 * Performs all steps required to generate random features for RBF-based
 * convolution kernels (FHTConv1d, GraphConv1d, ARD variants of GraphConv1d),
 * TOGETHER with the gradient.
 * It is assumed that caller has checked dimensions and they are all correct.
 *
 * ## Args:
 *
 * + `radem` The stacks of diagonal matrices used in
 * the transform. Must be of shape (3 x 1 x m * C) where m is
 * an integer that indicates the number of times we must repeat
 * the operation to generate the requested number of sampled frequencies.
 * + `reshapedX` Pointer to the first element of the array that will
 * be used for the convolution. A copy of this array is modified
 * rather than the original. Shape is (N x D x C). C must be
 * a power of 2.
 * + `copyBuffer` An array of the same size and shape as reshapedX into
 * which reshapedX can be copied. copyBuffer can then be modified in place
 * to generate the random features.
 * + `chiArr` A diagonal array that will be multiplied against the output
 * of the SORF operation. Of shape numFreqs.
 * + `outputArray` The output array. Of shape (N, 2 * numFreqs).
 * + `sigma` The lengthscale hyperparameter.
 * + `gradientArray` The array in which the gradient will be stored.
 * + `numThreads` The number of threads to use
 * + `reshapedDim0` The first dimension of reshapedX
 * + `reshapedDim1` The second dimension of reshapedX
 * + `reshapedDim2` The last dimension of reshapedX
 * + `numFreqs` The number of frequencies to sample. Must be <=
 * radem.shape[2].
 * + `rademShape2` The number of elements in one row of radem.
 *
 * ## Returns:
 * "error" if an error, "no_error" otherwise.
 */
const char *doubleConvRBFGrad_(int8_t *radem, double *reshapedX,
            double *copyBuffer, double *chiArr, double *outputArray,
            double *gradientArray, double sigma,
            int numThreads, int reshapedDim0,
            int reshapedDim1, int reshapedDim2,
            int numFreqs, int rademShape2)
{
    if (numThreads > reshapedDim0)
        numThreads = reshapedDim0;

    struct ThreadConvRBFDoubleArgs *th_args = (ThreadConvRBFDoubleArgs*)malloc(numThreads * sizeof(struct ThreadConvRBFDoubleArgs));
    if (th_args == NULL){
        PyErr_SetString(PyExc_ValueError, "System out of memory! RUN FOR YOUR LIFE!!!!");
        return "error";
    }
    std::vector<std::thread> threads(numThreads);
    int chunkSize = (reshapedDim0 + numThreads - 1) / numThreads;
    
    //We assume here that numFreqs is an integer multiple of reshapedDim2.
    //Caller must check this -- the Cython wrapper does.
    for (int i=0; i < numThreads; i++){
        th_args[i].startRow = i * chunkSize;
        th_args[i].endRow = (i + 1) * chunkSize;
        if (th_args[i].endRow > reshapedDim0)
            th_args[i].endRow = reshapedDim0;
        th_args[i].reshapedDim1 = reshapedDim1;
        th_args[i].reshapedDim2 = reshapedDim2;
        th_args[i].numFreqs = numFreqs;
        th_args[i].rademArray = radem;
        th_args[i].reshapedXArray = reshapedX;
        th_args[i].chiArr = chiArr;
        th_args[i].copyBuffer = copyBuffer;
        th_args[i].outputArray = outputArray;
        th_args[i].gradientArray = gradientArray;
        th_args[i].sigma = sigma;
        th_args[i].rademShape2 = rademShape2;
    }

    for (int i=0; i < numThreads; i++){
        threads[i] = std::thread(doubleThreadConvRBFGrad, &th_args[i]);
    }

    for (auto& th : threads)
        th.join();
    free(th_args);
    return "no_error";
}



/*!
 * # floatConvRBFGrad_
 * Performs all steps required to generate random features for RBF-based
 * convolution kernels (FHTConv1d, GraphConv1d, ARD variants of GraphConv1d),
 * TOGETHER with the gradient.
 * It is assumed that caller has checked dimensions and they are all correct.
 *
 * ## Args:
 *
 * + `radem` The stacks of diagonal matrices used in
 * the transform. Must be of shape (3 x 1 x m * C) where m is
 * an integer that indicates the number of times we must repeat
 * the operation to generate the requested number of sampled frequencies.
 * + `reshapedX` Pointer to the first element of the array that will
 * be used for the convolution. A copy of this array is modified
 * rather than the original. Shape is (N x D x C). C must be
 * a power of 2.
 * + `copyBuffer` An array of the same size and shape as reshapedX into
 * which reshapedX can be copied. copyBuffer can then be modified in place
 * to generate the random features.
 * + `chiArr` A diagonal array that will be multiplied against the output
 * of the SORF operation. Must be of shape numFreqs.
 * + `outputArray` The output array. Must be of shape (N, 2 * numFreqs).
 * + `gradientArray` The array in which the gradient will be stored.
 * + `sigma` The lengthscale hyperparameter.
 * + `numThreads` The number of threads to use
 * + `reshapedDim0` The first dimension of reshapedX
 * + `reshapedDim1` The second dimension of reshapedX
 * + `reshapedDim2` The last dimension of reshapedX
 * + `numFreqs` The number of frequencies to sample. Must be <=
 * radem.shape[2].
 * + `rademShape2` The number of elements in one row of radem.
 *
 * ## Returns:
 * "error" if an error, "no_error" otherwise.
 */
const char *floatConvRBFGrad_(int8_t *radem, float *reshapedX,
            float *copyBuffer, float *chiArr, double *outputArray,
            double *gradientArray, float sigma,
            int numThreads, int reshapedDim0,
            int reshapedDim1, int reshapedDim2,
            int numFreqs, int rademShape2)
{
    if (numThreads > reshapedDim0)
        numThreads = reshapedDim0;

    struct ThreadConvRBFFloatArgs *th_args = (ThreadConvRBFFloatArgs*)malloc(numThreads * sizeof(struct ThreadConvRBFFloatArgs));
    if (th_args == NULL){
        PyErr_SetString(PyExc_ValueError, "System out of memory!");
        return "error";
    }
    std::vector<std::thread> threads(numThreads);
    int chunkSize = (reshapedDim0 + numThreads - 1) / numThreads;
    
    //We assume here that numFreqs is an integer multiple of reshapedDim2.
    //Caller must check this -- the Cython wrapper does.
    for (int i=0; i < numThreads; i++){
        th_args[i].startRow = i * chunkSize;
        th_args[i].endRow = (i + 1) * chunkSize;
        if (th_args[i].endRow > reshapedDim0)
            th_args[i].endRow = reshapedDim0;
        th_args[i].reshapedDim1 = reshapedDim1;
        th_args[i].reshapedDim2 = reshapedDim2;
        th_args[i].numFreqs = numFreqs;
        th_args[i].rademArray = radem;
        th_args[i].reshapedXArray = reshapedX;
        th_args[i].chiArr = chiArr;
        th_args[i].copyBuffer = copyBuffer;
        th_args[i].outputArray = outputArray;
        th_args[i].gradientArray = gradientArray;
        th_args[i].sigma = sigma;
        th_args[i].rademShape2 = rademShape2;
    }

    for (int i=0; i < numThreads; i++){
        threads[i] = std::thread(floatThreadConvRBFGrad, &th_args[i]);
    }

    for (auto& th : threads)
        th.join();
    free(th_args);
    return "no_error";
}




/*!
 * # doubleThreadConvRBFGen
 *
 * Performs the RBF-based convolution kernel feature generation
 * process for the input, for one thread. reshapedX is split up into
 * num_threads chunks each with a start and end row.
 *
 * ## Args:
 * + `sharedArgs` A void pointer to a struct
 * containing pointers to the arrays needed to execute the
 * transform, the start and end rows etc.
 */
void *doubleThreadConvRBFGen(void *sharedArgs){
    struct ThreadConvRBFDoubleArgs *thArgs = (struct ThreadConvRBFDoubleArgs *)sharedArgs;
    int i, numRepeats, startPosition = 0;
    numRepeats = (thArgs->numFreqs +
            thArgs->reshapedDim2 - 1) / thArgs->reshapedDim2;

    for (i=0; i < numRepeats; i++){
        conv1dRademAndCopy<double>(thArgs->reshapedXArray,
                    thArgs->copyBuffer,
                    thArgs->rademArray, thArgs->startRow,
                    thArgs->endRow, thArgs->reshapedDim1, 
                    thArgs->reshapedDim2, startPosition);
        transformRows3D<double>(thArgs->copyBuffer, thArgs->startRow,
                    thArgs->endRow, thArgs->reshapedDim1, 
                    thArgs->reshapedDim2);
        conv1dMultiplyByRadem<double>(thArgs->copyBuffer,
                    thArgs->rademArray + thArgs->rademShape2,
                    thArgs->startRow, thArgs->endRow,
                    thArgs->reshapedDim1, thArgs->reshapedDim2,
                    startPosition);
        transformRows3D<double>(thArgs->copyBuffer, thArgs->startRow,
                    thArgs->endRow, thArgs->reshapedDim1, 
                    thArgs->reshapedDim2);
    
        conv1dMultiplyByRadem<double>(thArgs->copyBuffer,
                    thArgs->rademArray + 2 * thArgs->rademShape2,
                    thArgs->startRow, thArgs->endRow,
                    thArgs->reshapedDim1, thArgs->reshapedDim2,
                    startPosition);
        transformRows3D<double>(thArgs->copyBuffer, thArgs->startRow,
                    thArgs->endRow, thArgs->reshapedDim1, 
                    thArgs->reshapedDim2);

        RBFPostProcess<double>(thArgs->copyBuffer, thArgs->chiArr,
            thArgs->outputArray, thArgs->reshapedDim1,
            thArgs->reshapedDim2, thArgs->numFreqs, thArgs->startRow,
            thArgs->endRow, i);

        startPosition += thArgs->reshapedDim2;
    }
    return NULL;
}


/*!
 * # floatThreadConvRBFGen
 *
 * Performs the RBF-based convolution kernel feature generation
 * process for the input, for one thread. reshapedX is split up into
 * num_threads chunks each with a start and end row.
 *
 * ## Args:
 * + `sharedArgs` A void pointer to a struct
 * containing pointers to the arrays needed to execute the
 * transform, the start and end rows etc.
 */
void *floatThreadConvRBFGen(void *sharedArgs){
    struct ThreadConvRBFFloatArgs *thArgs = (struct ThreadConvRBFFloatArgs *)sharedArgs;
    int i, numRepeats, startPosition = 0;
    numRepeats = (thArgs->numFreqs +
            thArgs->reshapedDim2 - 1) / thArgs->reshapedDim2;

    for (i=0; i < numRepeats; i++){
        conv1dRademAndCopy<float>(thArgs->reshapedXArray,
                    thArgs->copyBuffer,
                    thArgs->rademArray, thArgs->startRow,
                    thArgs->endRow, thArgs->reshapedDim1, 
                    thArgs->reshapedDim2, startPosition);
        transformRows3D<float>(thArgs->copyBuffer, thArgs->startRow,
                    thArgs->endRow, thArgs->reshapedDim1, 
                    thArgs->reshapedDim2);
        conv1dMultiplyByRadem<float>(thArgs->copyBuffer,
                    thArgs->rademArray + thArgs->rademShape2,
                    thArgs->startRow, thArgs->endRow,
                    thArgs->reshapedDim1, thArgs->reshapedDim2,
                    startPosition);
        transformRows3D<float>(thArgs->copyBuffer, thArgs->startRow,
                    thArgs->endRow, thArgs->reshapedDim1, 
                    thArgs->reshapedDim2);
    
        conv1dMultiplyByRadem<float>(thArgs->copyBuffer,
                    thArgs->rademArray + 2 * thArgs->rademShape2,
                    thArgs->startRow, thArgs->endRow,
                    thArgs->reshapedDim1, thArgs->reshapedDim2,
                    startPosition);
        transformRows3D<float>(thArgs->copyBuffer, thArgs->startRow,
                    thArgs->endRow, thArgs->reshapedDim1, 
                    thArgs->reshapedDim2);
        RBFPostProcess<float>(thArgs->copyBuffer, thArgs->chiArr,
            thArgs->outputArray, thArgs->reshapedDim1,
            thArgs->reshapedDim2, thArgs->numFreqs, thArgs->startRow,
            thArgs->endRow, i);
        
        startPosition += thArgs->reshapedDim2;
    }
    return NULL;
}


/*!
 * # doubleThreadConvRBFGrad
 *
 * Performs the RBF-based convolution kernel feature generation
 * process for the input, for one thread. reshapedX is split up into
 * num_threads chunks each with a start and end row.
 *
 * ## Args:
 * + `sharedArgs` A void pointer to a struct
 * containing pointers to the arrays needed to execute the
 * transform, the start and end rows etc.
 */
void *doubleThreadConvRBFGrad(void *sharedArgs){
    struct ThreadConvRBFDoubleArgs *thArgs = (struct ThreadConvRBFDoubleArgs *)sharedArgs;
    int i, numRepeats, startPosition = 0;
    numRepeats = (thArgs->numFreqs +
            thArgs->reshapedDim2 - 1) / thArgs->reshapedDim2;

    for (i=0; i < numRepeats; i++){
        conv1dRademAndCopy<double>(thArgs->reshapedXArray,
                    thArgs->copyBuffer,
                    thArgs->rademArray, thArgs->startRow,
                    thArgs->endRow, thArgs->reshapedDim1, 
                    thArgs->reshapedDim2, startPosition);
        transformRows3D<double>(thArgs->copyBuffer, thArgs->startRow,
                    thArgs->endRow, thArgs->reshapedDim1, 
                    thArgs->reshapedDim2);
        conv1dMultiplyByRadem<double>(thArgs->copyBuffer,
                    thArgs->rademArray + thArgs->rademShape2,
                    thArgs->startRow, thArgs->endRow,
                    thArgs->reshapedDim1, thArgs->reshapedDim2,
                    startPosition);
        transformRows3D<double>(thArgs->copyBuffer, thArgs->startRow,
                    thArgs->endRow, thArgs->reshapedDim1, 
                    thArgs->reshapedDim2);
    
        conv1dMultiplyByRadem<double>(thArgs->copyBuffer,
                    thArgs->rademArray + 2 * thArgs->rademShape2,
                    thArgs->startRow, thArgs->endRow,
                    thArgs->reshapedDim1, thArgs->reshapedDim2,
                    startPosition);
        transformRows3D<double>(thArgs->copyBuffer, thArgs->startRow,
                    thArgs->endRow, thArgs->reshapedDim1, 
                    thArgs->reshapedDim2);
        RBFPostGrad<double>(thArgs->copyBuffer, thArgs->chiArr,
            thArgs->outputArray, thArgs->gradientArray, thArgs->reshapedDim1,
            thArgs->reshapedDim2, thArgs->numFreqs, thArgs->startRow, 
            thArgs->endRow, i, thArgs->sigma);

        startPosition += thArgs->reshapedDim2;
    }
    return NULL;
}


/*!
 * # floatThreadConvRBFGrad
 *
 * Performs the RBF-based convolution kernel feature generation
 * process for the input, for one thread. reshapedX is split up into
 * num_threads chunks each with a start and end row.
 *
 * ## Args:
 * + `sharedArgs` A void pointer to a struct
 * containing pointers to the arrays needed to execute the
 * transform, the start and end rows etc.
 */
void *floatThreadConvRBFGrad(void *sharedArgs){
    struct ThreadConvRBFFloatArgs *thArgs = (struct ThreadConvRBFFloatArgs *)sharedArgs;
    int i, numRepeats, startPosition = 0;
    numRepeats = (thArgs->numFreqs +
            thArgs->reshapedDim2 - 1) / thArgs->reshapedDim2;

    for (i=0; i < numRepeats; i++){
        conv1dRademAndCopy<float>(thArgs->reshapedXArray,
                    thArgs->copyBuffer,
                    thArgs->rademArray, thArgs->startRow,
                    thArgs->endRow, thArgs->reshapedDim1, 
                    thArgs->reshapedDim2, startPosition);
        transformRows3D<float>(thArgs->copyBuffer, thArgs->startRow,
                    thArgs->endRow, thArgs->reshapedDim1, 
                    thArgs->reshapedDim2);
        conv1dMultiplyByRadem<float>(thArgs->copyBuffer,
                    thArgs->rademArray + thArgs->rademShape2,
                    thArgs->startRow, thArgs->endRow,
                    thArgs->reshapedDim1, thArgs->reshapedDim2,
                    startPosition);
        transformRows3D<float>(thArgs->copyBuffer, thArgs->startRow,
                    thArgs->endRow, thArgs->reshapedDim1, 
                    thArgs->reshapedDim2);
    
        conv1dMultiplyByRadem<float>(thArgs->copyBuffer,
                    thArgs->rademArray + 2 * thArgs->rademShape2,
                    thArgs->startRow, thArgs->endRow,
                    thArgs->reshapedDim1, thArgs->reshapedDim2,
                    startPosition);
        transformRows3D<float>(thArgs->copyBuffer, thArgs->startRow,
                    thArgs->endRow, thArgs->reshapedDim1, 
                    thArgs->reshapedDim2);
        RBFPostGrad<float>(thArgs->copyBuffer, thArgs->chiArr,
            thArgs->outputArray, thArgs->gradientArray, thArgs->reshapedDim1,
            thArgs->reshapedDim2, thArgs->numFreqs, thArgs->startRow,
            thArgs->endRow, i, thArgs->sigma);
        
        startPosition += thArgs->reshapedDim2;
    }
    return NULL;
}

/*!
 * # RBFPostProcess
 *
 * Performs the last steps in RBF-based convolution kernel feature
 * generation.
 *
 * ## Args:
 * + `reshapedX` Pointer to the first element of the array that has been
 * used for the convolution. Shape is (N x D x C). C must be
 * a power of 2.
 * + `chiArr` Pointer to the first element of chiArr, a diagonal array
 * that will be multipled against reshapedX.
 * + `outputArray` A pointer to the first element of the array in which
 * the output will be stored.
 * + `reshapedDim1` The second dimension of reshapedX
 * + `reshapedDim2` The last dimension of reshapedX
 * + `numFreqs` The number of frequencies to sample.
 * + `startRow` The first row of the input to work on
 * + `endRow` The last row of the input to work on
 * + `repeatNum` The repeat number
 */
template <typename T>
void RBFPostProcess(T reshapedX[], T chiArr[],
        double *outputArray, int reshapedDim1,
        int reshapedDim2, int numFreqs,
        int startRow, int endRow, int repeatNum){
    int i, j, k, lenOutputRow, outputStart;
    T prodVal;
    double *xOut;
    T *xIn, *chiIn;
    int endPosition, lenInputRow = reshapedDim1 * reshapedDim2;

    outputStart = repeatNum * reshapedDim2;

    //NOTE: MIN is defined in the header.
    endPosition = MIN(numFreqs, (repeatNum + 1) * reshapedDim2);
    endPosition -= outputStart;
    lenOutputRow = 2 * numFreqs;
    chiIn = chiArr + outputStart;
    xIn = reshapedX + startRow * lenInputRow;

    for (i=startRow; i < endRow; i++){
        for (k=0; k < reshapedDim1; k++){
            xOut = outputArray + i * lenOutputRow + 2 * outputStart;
            for (j=0; j < endPosition; j++){
                prodVal = xIn[j] * chiIn[j];
                *xOut += cos(prodVal);
                xOut++;
                *xOut += sin(prodVal);
                xOut++;
            }
            xIn += reshapedDim2;
        }
    }
}





/*!
 * # RBFPostGrad
 *
 * Performs the last steps in RBF-based convolution kernel feature
 * generation, while additionally calculating the gradient w/r/t
 * the lengthscale.
 *
 * ## Args:
 * + `reshapedX` Pointer to the first element of the array that has been
 * used for the convolution. Shape is (N x D x C). C must be
 * a power of 2.
 * + `chiArr` Pointer to the first element of chiArr, a diagonal array
 * that will be multipled against reshapedX.
 * + `outputArray` A pointer to the first element of the array in which
 * the output will be stored.
 * + `gradientArray` A pointer to the first element of the array in which
 * the gradient will be stored.
 * + `reshapedDim1` The second dimension of reshapedX
 * + `reshapedDim2` The last dimension of reshapedX
 * + `numFreqs` The number of frequencies to sample.
 * + `startRow` The first row of the input to work on
 * + `endRow` The last row of the input to work on
 * + `repeatNum` The repeat number
 * + `sigma` The lengthscale hyperparameter
 */
template <typename T>
void RBFPostGrad(T reshapedX[], T chiArr[],
        double *outputArray, double *gradientArray,
        int reshapedDim1, int reshapedDim2,
        int numFreqs, int startRow, int endRow,
        int repeatNum, T sigma){
    int i, j, k, lenOutputRow, outputStart;
    T prodVal, gradVal, cosVal, sinVal;
    double *xOut, *gradOut;
    T *xIn, *chiIn;
    int endPosition, lenInputRow = reshapedDim1 * reshapedDim2;

    outputStart = repeatNum * reshapedDim2;

    //NOTE: MIN is defined in the header.
    endPosition = MIN(numFreqs, (repeatNum + 1) * reshapedDim2);
    endPosition -= outputStart;

    lenOutputRow = 2 * numFreqs;
    chiIn = chiArr + outputStart;

    for (i=startRow; i < endRow; i++){
        xIn = reshapedX + i * lenInputRow;
        xOut = outputArray + i * lenOutputRow + 2 * outputStart;
        gradOut = gradientArray + i * lenOutputRow + 2 * outputStart;

        for (k=0; k < reshapedDim1; k++){
            for (j=0; j < endPosition; j++){
                gradVal = xIn[j] * chiIn[j];
                prodVal = gradVal * sigma;
                cosVal = cos(prodVal);
                sinVal = sin(prodVal);
                xOut[2*j] += cosVal;
                xOut[2*j+1] += sinVal;
                gradOut[2*j] += -sinVal * gradVal;
                gradOut[2*j+1] += cosVal * gradVal;
            }
            xIn += reshapedDim2;
        }
    }
}
