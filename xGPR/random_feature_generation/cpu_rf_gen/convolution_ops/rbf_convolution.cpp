/*!
 * # rbf_convolution.cpp
 *
 * This module performs operations unique to the RBF-based convolution
 * kernels in xGPR.
 */
#include <thread>
#include <vector>
#include <math.h>
#include "rbf_convolution.h"
#include "../shared_fht_functions/hadamard_transforms.h"
#include "../shared_fht_functions/shared_rfgen_ops.h"






/*!
 * # convRBFFeatureGen_
 * Performs all steps required to generate random features for RBF-based
 * convolution kernels.
 *
 * ## Args:
 *
 * + `inputArr` The (N x D x C) array containing the input data.
 * + `outputArr` The (N x R) output array, where R = 2 * F and F is numFreqs.
 * + `radem` The (3 x 1 x M) array of int8_t diagonal matrices, where M is
 * some integer multiple of the smallest power of 2 > C and is > numFreqs.
 * + `chiArr` The (F) shape numpy diagonal matrix by which the results are scaled.
 * + `seqlengths` An (N) shape numpy array of sequence lengths (to exclude zero
 * padding).
 * + `convWidth` The width of the convolution kernel.
 * + `scalingType` One of 0, 1 or 2; indicates the type of scaling. These
 * are defined in the header.
 * + `numThreads` The number of threads to use.
 */
template <typename T>
int convRBFFeatureGen_(nb::ndarray<T, nb::shape<-1,-1,-1>, nb::device::cpu, nb::c_contig> inputArr,
        nb::ndarray<double, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> outputArr,
        nb::ndarray<int8_t, nb::shape<3,1,-1>, nb::device::cpu, nb::c_contig> radem,
        nb::ndarray<T, nb::shape<-1>, nb::device::cpu, nb::c_contig> chiArr,
        nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> seqlengths,
        int convWidth, int scalingType, int numThreads) {

    // Perform safety checks. Any exceptions thrown here are handed off to Python
    // by the Nanobind wrapper. We do not expect the user to see these because
    // the Python code will always ensure inputs are correct -- these are a failsafe
    // -- so we do not need to provide detailed exception messages here.
    int zDim0 = inputArr.shape(0);
    size_t numRffs = outputArr.shape(1);
    size_t numFreqs = chiArr.shape(0);
    double scalingTerm = std::sqrt(1.0 / static_cast<double>(numFreqs));

    T *inputPtr = static_cast<T*>(inputArr.data());
    double *outputPtr = static_cast<double*>(outputArr.data());
    T *chiPtr = static_cast<T*>(chiArr.data());
    int8_t *rademPtr = static_cast<int8_t*>(radem.data());
    int32_t *seqlengthsPtr = static_cast<int32_t*>(seqlengths.data());

    if (inputArr.shape(0) == 0 || outputArr.shape(0) != inputArr.shape(0))
        throw std::runtime_error("no datapoints");
    if (numRffs < 2 || (numRffs & 1) != 0)
        throw std::runtime_error("last dim of output must be even number");
    if ( (2 * numFreqs) != numRffs || numFreqs > radem.shape(2) )
        throw std::runtime_error("incorrect number of rffs and or freqs.");

    if (seqlengths.shape(0) != inputArr.shape(0))
        throw std::runtime_error("wrong array sizes");
    if (static_cast<int>(inputArr.shape(1)) < convWidth || convWidth <= 0)
        throw std::runtime_error("invalid conv_width");

    double expectedNFreq = static_cast<double>(convWidth * inputArr.shape(2));
    expectedNFreq = MAX(expectedNFreq, 2);
    double log2Freqs = std::log2(expectedNFreq);
    log2Freqs = std::ceil(log2Freqs);
    int paddedBufferSize = std::pow(2, log2Freqs);

    if (radem.shape(2) % paddedBufferSize != 0)
        throw std::runtime_error("incorrect number of rffs and or freqs.");


    int32_t minSeqLength = 2147483647, maxSeqLength = 0;
    for (size_t i=0; i < seqlengths.shape(0); i++) {
        if (seqlengths(i) > maxSeqLength)
            maxSeqLength = seqlengths(i);
        if (seqlengths(i) < minSeqLength)
            minSeqLength = seqlengths(i);
    }

    if (maxSeqLength > static_cast<int32_t>(inputArr.shape(1)) || minSeqLength < convWidth) {
        throw std::runtime_error("All sequence lengths must be >= conv width and < "
                "array size.");
    }



    if (numThreads > zDim0)
        numThreads = zDim0;

    std::vector<std::thread> threads(numThreads);
    int startRow, endRow;
    int chunkSize = (zDim0 + numThreads - 1) / numThreads;
    
    for (int i=0; i < numThreads; i++) {
        startRow = i * chunkSize;
        endRow = (i + 1) * chunkSize;
        if (endRow > zDim0)
            endRow = zDim0;

        threads[i] = std::thread(&allInOneConvRBFGen<T>, inputPtr,
                rademPtr, chiPtr, outputPtr,
                seqlengthsPtr, inputArr.shape(1), inputArr.shape(2),
                numFreqs, radem.shape(2), startRow, endRow, convWidth,
                paddedBufferSize, scalingTerm, scalingType);
    }

    for (auto& th : threads)
        th.join();

    return 0;
}
//Instantiate the templates the wrapper will need to access.
template int convRBFFeatureGen_<double>(nb::ndarray<double, nb::shape<-1,-1,-1>, nb::device::cpu, nb::c_contig> inputArr,
        nb::ndarray<double, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> outputArr,
        nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cpu, nb::c_contig> radem,
        nb::ndarray<double, nb::shape<-1>, nb::device::cpu, nb::c_contig> chiArr,
        nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> seqlengths,
        int convWidth, int scalingType, int numThreads);
template int convRBFFeatureGen_<float>(nb::ndarray<float, nb::shape<-1,-1,-1>, nb::device::cpu, nb::c_contig> inputArr,
        nb::ndarray<double, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> outputArr,
        nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cpu, nb::c_contig> radem,
        nb::ndarray<float, nb::shape<-1>, nb::device::cpu, nb::c_contig> chiArr,
        nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> seqlengths,
        int convWidth, int scalingType, int numThreads);





/*!
 * # convRBFGrad_
 * Performs all steps required to generate random features for RBF-based
 * convolution kernels (FHTConv1d, GraphConv1d, ARD variants of GraphConv1d),
 * TOGETHER with the gradient.
 * It is assumed that caller has checked dimensions and they are all correct.
 *
 * ## Args:
 *
 * + `inputArr` The (N x D x C) array containing the input data.
 * + `outputArr` The (N x R) output array, where R = 2 * F and F is numFreqs.
 * + `radem` The (3 x 1 x M) array of int8_t diagonal matrices, where M is
 * some integer multiple of the smallest power of 2 > C and is > numFreqs.
 * + `chiArr` The (F) shape numpy diagonal matrix by which the results are scaled.
 * + `seqlengths` An (N) shape numpy array of sequence lengths (to exclude zero
 * padding).
 * + `gradArr` The (N x R x 1) array containing the gradient.
 * + `convWidth` The width of the convolution kernel.
 * + `sigma` The sigma hyperparameter.
 * + `scalingType` One of 0, 1 or 2; indicates the type of scaling. These
 * are defined in the header.
 * + `numThreads` The number of threads to use.
 */
template <typename T>
int convRBFGrad_(nb::ndarray<T, nb::shape<-1,-1,-1>, nb::device::cpu, nb::c_contig> inputArr,
        nb::ndarray<double, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> outputArr,
        nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cpu, nb::c_contig> radem,
        nb::ndarray<T, nb::shape<-1>, nb::device::cpu, nb::c_contig> chiArr,
        nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> seqlengths,
        nb::ndarray<double, nb::shape<-1,-1,1>, nb::device::cpu, nb::c_contig> gradArr,
        double sigma, int convWidth, int scalingType, int numThreads) {

    // Perform safety checks. Any exceptions thrown here are handed off to Python
    // by the Nanobind wrapper. We do not expect the user to see these because
    // the Python code will always ensure inputs are correct -- these are a failsafe
    // -- so we do not need to provide detailed exception messages here.
    int zDim0 = inputArr.shape(0);
    size_t numRffs = outputArr.shape(1);
    size_t numFreqs = chiArr.shape(0);
    double scalingTerm = std::sqrt(1.0 / static_cast<double>(numFreqs));

    T *inputPtr = static_cast<T*>(inputArr.data());
    double *outputPtr = static_cast<double*>(outputArr.data());
    T *chiPtr = static_cast<T*>(chiArr.data());
    int8_t *rademPtr = static_cast<int8_t*>(radem.data());
    int32_t *seqlengthsPtr = static_cast<int32_t*>(seqlengths.data());
    double *gradientPtr = static_cast<double*>(gradArr.data());

    if (inputArr.shape(0) == 0 || outputArr.shape(0) != inputArr.shape(0))
        throw std::runtime_error("no datapoints");
    if (numRffs < 2 || (numRffs & 1) != 0)
        throw std::runtime_error("last dim of output must be even number");
    if ( (2 * numFreqs) != numRffs || numFreqs > radem.shape(2) )
        throw std::runtime_error("incorrect number of rffs and or freqs.");

    if (seqlengths.shape(0) != inputArr.shape(0))
        throw std::runtime_error("wrong array sizes");
    if (static_cast<int>(inputArr.shape(1)) < convWidth || convWidth <= 0)
        throw std::runtime_error("invalid conv_width");

    if (gradArr.shape(0) != outputArr.shape(0) || gradArr.shape(1) != outputArr.shape(1))
        throw std::runtime_error("wrong array sizes");

    double expectedNFreq = static_cast<double>(convWidth * inputArr.shape(2));
    expectedNFreq = MAX(expectedNFreq, 2);
    double log2Freqs = std::log2(expectedNFreq);
    log2Freqs = std::ceil(log2Freqs);
    int paddedBufferSize = std::pow(2, log2Freqs);

    if (radem.shape(2) % paddedBufferSize != 0)
        throw std::runtime_error("incorrect number of rffs and or freqs.");


    int32_t minSeqLength = 2147483647, maxSeqLength = 0;
    for (size_t i=0; i < seqlengths.shape(0); i++) {
        if (seqlengths(i) > maxSeqLength)
            maxSeqLength = seqlengths(i);
        if (seqlengths(i) < minSeqLength)
            minSeqLength = seqlengths(i);
    }

    if (maxSeqLength > static_cast<int32_t>(inputArr.shape(1)) || minSeqLength < convWidth) {
        throw std::runtime_error("All sequence lengths must be >= conv width and < "
                "array size.");
    }



    if (numThreads > zDim0)
        numThreads = zDim0;

    std::vector<std::thread> threads(numThreads);
    int startRow, endRow;
    int chunkSize = (zDim0 + numThreads - 1) / numThreads;
    
    for (int i=0; i < numThreads; i++) {
        startRow = i * chunkSize;
        endRow = (i + 1) * chunkSize;
        if (endRow > zDim0)
            endRow = zDim0;

        threads[i] = std::thread(&allInOneConvRBFGrad<T>, inputPtr,
                rademPtr, chiPtr, outputPtr, seqlengthsPtr, gradientPtr,
                inputArr.shape(1), inputArr.shape(2),
                numFreqs, radem.shape(2), startRow,
                endRow, convWidth, paddedBufferSize,
                scalingTerm, scalingType, sigma);

    }

    for (auto& th : threads)
        th.join();

    return 0;
}
//Instantiate templates for use by wrapper.
template int convRBFGrad_<double>(nb::ndarray<double, nb::shape<-1,-1,-1>, nb::device::cpu, nb::c_contig> inputArr,
        nb::ndarray<double, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> outputArr,
        nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cpu, nb::c_contig> radem,
        nb::ndarray<double, nb::shape<-1>, nb::device::cpu, nb::c_contig> chiArr,
        nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> seqlengths,
        nb::ndarray<double, nb::shape<-1,-1,1>, nb::device::cpu, nb::c_contig> gradArr,
        double sigma, int convWidth, int scalingType, int numThreads);
template int convRBFGrad_<float>(nb::ndarray<float, nb::shape<-1,-1,-1>, nb::device::cpu, nb::c_contig> inputArr,
        nb::ndarray<double, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> outputArr,
        nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cpu, nb::c_contig> radem,
        nb::ndarray<float, nb::shape<-1>, nb::device::cpu, nb::c_contig> chiArr,
        nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> seqlengths,
        nb::ndarray<double, nb::shape<-1,-1,1>, nb::device::cpu, nb::c_contig> gradArr,
        double sigma, int convWidth, int scalingType, int numThreads);




/*!
 * # allInOneConvRBFGen
 *
 * Performs the RBF-based convolution kernel feature generation
 * process for the input, for one thread.
 */
template <typename T>
void *allInOneConvRBFGen(T xdata[], int8_t *rademArray, T chiArr[],
        double *outputArray, int32_t *seqlengths, int dim1, int dim2,
        int numFreqs, int rademShape2, int startRow, int endRow,
        int convWidth, int paddedBufferSize,
        double scalingTerm, int scalingType) {

    int numKmers;
    int32_t seqlength;
    int numRepeats = (numFreqs + paddedBufferSize - 1) / paddedBufferSize;
    double rowScaler;
    //Notice that we don't have error handling here...very naughty. Out of
    //memory should be extremely rare since we are only allocating memory
    //for one row of the convolution. TODO: add error handling here.
    T *copyBuffer = new T[paddedBufferSize];
    T *xElement;

    for (int i=startRow; i < endRow; i++) {
        seqlength = seqlengths[i];
        numKmers = seqlength - convWidth + 1;
        switch (scalingType) {
            case SQRT_CONVOLUTION_SCALING:
                rowScaler = scalingTerm / std::sqrt( (double)numKmers);
                break;
            case FULL_CONVOLUTION_SCALING:
                rowScaler = scalingTerm / (double)numKmers;
                break;
            default:
               rowScaler = scalingTerm;
              break; 
        }

        for (int j=0; j < numKmers; j++) {
            int repeatPosition = 0;
            xElement = xdata + i * dim1 * dim2 + j * dim2;

            for (int k=0; k < numRepeats; k++) {
                for (int m=0; m < (convWidth * dim2); m++)
                    copyBuffer[m] = xElement[m];
                for (int m=(convWidth * dim2); m < paddedBufferSize; m++)
                    copyBuffer[m] = 0;

                singleVectorSORF(copyBuffer, rademArray, repeatPosition,
                        rademShape2, paddedBufferSize);
                singleVectorRBFPostProcess(copyBuffer, chiArr, outputArray,
                        paddedBufferSize, numFreqs, i, k, rowScaler);
                repeatPosition += paddedBufferSize;
            }
        }
    }

    delete[] copyBuffer;

    return NULL;
}







/*!
 * # allInOneConvRBFGrad
 *
 * Performs the RBF-based convolution kernel feature generation
 * process for the input, for one thread, and calculates the
 * gradient, which is stored in a separate array.
 */
template <typename T>
void *allInOneConvRBFGrad(T xdata[], int8_t *rademArray, T chiArr[],
        double *outputArray, int32_t *seqlengths, double *gradientArray,
        int dim1, int dim2, int numFreqs, int rademShape2, int startRow,
        int endRow, int convWidth, int paddedBufferSize,
        double scalingTerm, int scalingType, T sigma) {

    int numKmers;
    int32_t seqlength;
    int numRepeats = (numFreqs + paddedBufferSize - 1) / paddedBufferSize;
    double rowScaler;
    //Notice that we don't have error handling here...very naughty. Out of
    //memory should be extremely rare since we are only allocating memory
    //for one row of the convolution. TODO: add error handling here.
    T *copyBuffer = new T[paddedBufferSize];
    T *xElement;

    for (int i=startRow; i < endRow; i++) {
        seqlength = seqlengths[i];
        numKmers = seqlength - convWidth + 1;
        switch (scalingType) {
            case SQRT_CONVOLUTION_SCALING:
                rowScaler = scalingTerm / std::sqrt(
                        static_cast<double>(numKmers));
                break;
            case FULL_CONVOLUTION_SCALING:
                rowScaler = scalingTerm / static_cast<double>(numKmers);
                break;
            default:
                rowScaler = scalingTerm;
              break;
        }

        for (int j=0; j < numKmers; j++) {
            int repeatPosition = 0;
            xElement = xdata + i * dim1 * dim2 + j * dim2;

            for (int k=0; k < numRepeats; k++) {
                for (int m=0; m < (convWidth * dim2); m++)
                    copyBuffer[m] = xElement[m];
                for (int m=(convWidth * dim2); m < paddedBufferSize; m++)
                    copyBuffer[m] = 0;

                singleVectorSORF(copyBuffer, rademArray, repeatPosition,
                        rademShape2, paddedBufferSize);
                singleVectorRBFPostGrad(copyBuffer, chiArr, outputArray,
                        gradientArray, sigma, paddedBufferSize, numFreqs,
                        i, k, rowScaler);
                repeatPosition += paddedBufferSize;
            }
        }
    }
    delete[] copyBuffer;

    return NULL;
}
