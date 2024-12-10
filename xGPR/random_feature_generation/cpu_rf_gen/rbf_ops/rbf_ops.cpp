/*!
 * # rbf_ops.cpp
 *
 * This module performs all major steps involved in feature generation for
 * RBF-type kernels, which includes RBF, Matern, Cauchy, MiniARD.
 */
#include <math.h>
#include <vector>
#include <thread>
#include "rbf_ops.h"
#include "../shared_fht_functions/hadamard_transforms.h"
#include "../shared_fht_functions/shared_rfgen_ops.h"

namespace nb = nanobind;



/*!
 * # rbfFeatureGen_
 *
 * Generates features for the input array.
 *
 * ## Args:
 *
 * + `inputArr` A numpy array of shape (N x C).
 * + `outputArr` A numpy array of shape (N x R),
 * where R is the number of RFFs and is 2x numFreqs;
 * + `radem` A numpy stack of diagonal matrices of type int8_t
 * of shape (3 x 1 x M) where M is the smallest power of 2 > numFreqs.
 * + `chiArr` A numpy array of shape (numFreqs)
 * + `numThreads` The number of threads to use.
 * + `fitIntercept` If True, a y-intercept will be fitted.
 */
template <typename T>
int rbfFeatureGen_(nb::ndarray<T, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> inputArr,
        nb::ndarray<double, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> outputArr,
        nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cpu, nb::c_contig> radem,
        nb::ndarray<T, nb::shape<-1>, nb::device::cpu, nb::c_contig> chiArr,
        int numThreads, bool fitIntercept) {
    // Perform safety checks. Any exceptions thrown here are handed off to Python
    // by the Nanobind wrapper. We do not expect the user to see these because
    // the Python code will always ensure inputs are correct -- these are a failsafe
    // -- so we do not need to provide detailed exception messages here.
    int zDim0 = inputArr.shape(0);
    int zDim1 = inputArr.shape(1);
    size_t numRffs = outputArr.shape(1);
    size_t numFreqs = chiArr.shape(0);
    double numFreqsFlt = numFreqs;

    T *inputPtr = static_cast<T*>(inputArr.data());
    double *outputPtr = static_cast<double*>(outputArr.data());
    T *chiPtr = static_cast<T*>(chiArr.data());
    int8_t *rademPtr = static_cast<int8_t*>(radem.data());

    if (inputArr.shape(0) == 0 || outputArr.shape(0) != inputArr.shape(0))
        throw std::runtime_error("no datapoints");
    if (numRffs < 2 || (numRffs & 1) != 0)
        throw std::runtime_error("last dim of output must be even number");
    if ( (2 * numFreqs) != numRffs || numFreqs > radem.shape(2) )
        throw std::runtime_error("incorrect number of rffs and or freqs.");

    double expectedNFreq = (zDim1 > 2) ? static_cast<double>(zDim1) : 2.0;
    double log2Freqs = std::log2(expectedNFreq);
    log2Freqs = std::ceil(log2Freqs);
    int paddedBufferSize = std::pow(2, log2Freqs);

    if (radem.shape(2) % paddedBufferSize != 0)
        throw std::runtime_error("incorrect number of rffs and or freqs.");

    T rbfNormConstant;

    if (fitIntercept)
        rbfNormConstant = std::sqrt(1.0 / (numFreqsFlt - 0.5));
    else
        rbfNormConstant = std::sqrt(1.0 / numFreqsFlt);


    if (numThreads > zDim0)
        numThreads = zDim0;

    std::vector<std::thread> threads(numThreads);
    int startPosition, endPosition;

    int chunkSize = (zDim0 + numThreads - 1) / numThreads;

    for (int i=0; i < numThreads; i++) {
        startPosition = i * chunkSize;
        endPosition = (i + 1) * chunkSize;
        if (endPosition > zDim0)
            endPosition = zDim0;

        threads[i] = std::thread(&allInOneRBFGen<T>, inputPtr,
                rademPtr, chiPtr, outputPtr, inputArr.shape(1), numFreqs,
                radem.shape(2), startPosition, endPosition,
                paddedBufferSize, rbfNormConstant);
    }

    for (auto& th : threads)
        th.join();
    return 0;
}
//Explicitly instantiate so wrapper can use.
template int rbfFeatureGen_<double>(nb::ndarray<double, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> inputArr,
        nb::ndarray<double, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> outputArr,
        nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cpu, nb::c_contig> radem,
        nb::ndarray<double, nb::shape<-1>, nb::device::cpu, nb::c_contig> chiArr,
        int numThreads, bool fitIntercept);
template int rbfFeatureGen_<float>(nb::ndarray<float, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> inputArr,
        nb::ndarray<double, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> outputArr,
        nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cpu, nb::c_contig> radem,
        nb::ndarray<float, nb::shape<-1>, nb::device::cpu, nb::c_contig> chiArr,
        int numThreads, bool fitIntercept);




/*!
 * # rbfGrad_
 *
 * Generates features and the gradient w/r/t sigma.
 *
 * ## Args:
 *
 * + `inputArr` A numpy array of shape (N x C).
 * + `outputArr` A numpy array of shape (N x R),
 * where R is the number of RFFs and is 2x numFreqs;
 * + `gradArr` A numpy array of shape (N x R x 1),
 * where R is the number of RFFs and is 2x numFreqs;
 * + `radem` A numpy stack of diagonal matrices of type int8_t
 * of shape (3 x 1 x M) where M is the smallest power of 2 > numFreqs.
 * + `chiArr` A numpy array of shape (numFreqs)
 * + `sigma` The sigma hyperparameter
 * + `numThreads` The number of threads to use.
 * + `fitIntercept` If True, a y-intercept will be fitted.
 */
template <typename T>
int rbfGrad_(nb::ndarray<T, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> inputArr,
        nb::ndarray<double, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> outputArr,
        nb::ndarray<double, nb::shape<-1,-1,1>, nb::device::cpu, nb::c_contig> gradArr,
        nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cpu, nb::c_contig> radem,
        nb::ndarray<T, nb::shape<-1>, nb::device::cpu, nb::c_contig> chiArr,
        float sigma, int numThreads, bool fitIntercept) {
    // Perform safety checks. Any exceptions thrown here are handed off to Python
    // by the Nanobind wrapper. We do not expect the user to see these because
    // the Python code will always ensure inputs are correct -- these are a failsafe
    // -- so we do not need to provide detailed exception messages here.
    int zDim0 = inputArr.shape(0);
    int zDim1 = inputArr.shape(1);
    size_t numRffs = outputArr.shape(1);
    size_t numFreqs = chiArr.shape(0);
    double numFreqsFlt = numFreqs;

    T *inputPtr = static_cast<T*>(inputArr.data());
    double *outputPtr = static_cast<double*>(outputArr.data());
    double *gradientPtr = static_cast<double*>(gradArr.data());
    T *chiPtr = static_cast<T*>(chiArr.data());
    int8_t *rademPtr = static_cast<int8_t*>(radem.data());

    if (inputArr.shape(0) == 0 || outputArr.shape(0) != inputArr.shape(0))
        throw std::runtime_error("no datapoints");
    if (numRffs < 2 || (numRffs & 1) != 0)
        throw std::runtime_error("last dim of output must be even number");
    if ( (2 * numFreqs) != numRffs || numFreqs > radem.shape(2) )
        throw std::runtime_error("incorrect number of rffs and or freqs.");
    if (gradArr.shape(0) != outputArr.shape(0) || gradArr.shape(1) != outputArr.shape(1))
        throw std::runtime_error("Wrong array sizes.");

    double expectedNFreq = (zDim1 > 2) ? static_cast<double>(zDim1) : 2.0;
    double log2Freqs = std::log2(expectedNFreq);
    log2Freqs = std::ceil(log2Freqs);
    int paddedBufferSize = std::pow(2, log2Freqs);

    if (radem.shape(2) % paddedBufferSize != 0)
        throw std::runtime_error("incorrect number of rffs and or freqs.");

    T rbfNormConstant;

    if (fitIntercept)
        rbfNormConstant = std::sqrt(1.0 / (numFreqsFlt - 0.5));
    else
        rbfNormConstant = std::sqrt(1.0 / numFreqsFlt);
    
    
    
    if (numThreads > zDim0)
        numThreads = zDim0;

    std::vector<std::thread> threads(numThreads);
    int startPosition, endPosition;
    int chunkSize = (zDim0 + numThreads - 1) / numThreads;

    for (int i=0; i < numThreads; i++) {
        startPosition = i * chunkSize;
        endPosition = (i + 1) * chunkSize;
        if (endPosition > zDim0)
            endPosition = zDim0;

        threads[i] = std::thread(&allInOneRBFGrad<T>, inputPtr,
                rademPtr, chiPtr, outputPtr,
                gradientPtr, inputArr.shape(1),
                numFreqs, radem.shape(2), startPosition,
                endPosition, paddedBufferSize,
                rbfNormConstant, sigma);
    }

    for (auto& th : threads)
        th.join();
    return 0;
}
//Explicitly instantiate for external use.
template int rbfGrad_<double>(nb::ndarray<double, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> inputArr,
        nb::ndarray<double, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> outputArr,
        nb::ndarray<double, nb::shape<-1,-1,1>, nb::device::cpu, nb::c_contig> gradArr,
        nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cpu, nb::c_contig> radem,
        nb::ndarray<double, nb::shape<-1>, nb::device::cpu, nb::c_contig> chiArr,
        float sigma, int numThreads, bool fitIntercept);
template int rbfGrad_<float>(nb::ndarray<float, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> inputArr,
        nb::ndarray<double, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> outputArr,
        nb::ndarray<double, nb::shape<-1,-1,1>, nb::device::cpu, nb::c_contig> gradArr,
        nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cpu, nb::c_contig> radem,
        nb::ndarray<float, nb::shape<-1>, nb::device::cpu, nb::c_contig> chiArr,
        float sigma, int numThreads, bool fitIntercept);




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
        double scalingTerm) {
    int numRepeats = (numFreqs + paddedBufferSize - 1) / paddedBufferSize;
    //Notice that we don't have error handling here...very naughty. Out of
    //memory should be extremely rare since we are only allocating memory
    //for one row of the convolution. TODO: add error handling here.
    T *copyBuffer = new T[paddedBufferSize];
    T *xElement;

    for (int i=startRow; i < endRow; i++) {

        int repeatPosition = 0;
        xElement = xdata + i * dim1;

        for (int k=0; k < numRepeats; k++) {
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
        double scalingTerm, T sigma) {
    int numRepeats = (numFreqs + paddedBufferSize - 1) / paddedBufferSize;
    //Notice that we don't have error handling here...very naughty. Out of
    //memory should be extremely rare since we are only allocating memory
    //for one row of the convolution. TODO: add error handling here.
    T *copyBuffer = new T[paddedBufferSize];
    T *xElement;

    for (int i=startRow; i < endRow; i++) {
        int repeatPosition = 0;
        xElement = xdata + i * dim1;

        for (int k=0; k < numRepeats; k++) {
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
