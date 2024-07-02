/*!
 * # transform_functions.cpp
 *
 * Performs fast Hadamard transforms, SORF and SRHT operations on a variety of different
 * array shapes.
 */
#include <vector>
#include <thread>
#include <stdexcept>
#include "transform_functions.h"
#include "../shared_fht_functions/hadamard_transforms.h"
#include "../shared_fht_functions/shared_rfgen_ops.h"

namespace nb = nanobind;


/*!
 * # fastHadamard3dArray_
 *
 * Performs an unnormalized Hadamard transform along the last
 * dimension of an input 3d array. The transform is performed
 * in place.
 *
 * ## Args:
 * + `inputArr` A nanobind reference to a numpy 3d array of shape
 * (N x D x C), where C must be a power of 2.
 * + `numThreads` The number of threads to use.
 */
template <typename T>
int fastHadamard3dArray_(nb::ndarray<T, nb::shape<-1,-1,-1>,
                       nb::device::cpu, nb::c_contig> inputArr, int numThreads){

    // Perform safety checks. Any exceptions thrown here are handed off to Python
    // by the Nanobind wrapper. We do not expect the user to see these because
    // the Python code will always ensure inputs are correct -- these are a failsafe
    // -- so we do not need to provide detailed exception messages here.
    int zDim0 = inputArr.shape(0);
    int zDim1 = inputArr.shape(1);
    int zDim2 = inputArr.shape(2);
    T *inputPtr = static_cast<T*>(inputArr.data());

    if (inputArr.shape(0) == 0)
        throw std::runtime_error("no datapoints");
    if (inputArr.shape(2) < 2)
        throw std::runtime_error("last dim not power of 2 > 1");
    if ((zDim2 & (zDim2 - 1)) != 0)
        throw std::runtime_error("last dim not power of 2");


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
        threads[i] = std::thread(&ThreadTransformRows3D<T>, inputPtr, startPosition,
                                endPosition, zDim1, zDim2);
    }

    for (auto& th : threads)
        th.join();
    return 0;
}
template int fastHadamard3dArray_<double>(nb::ndarray<double, nb::shape<-1,-1,-1>,
                       nb::device::cpu, nb::c_contig> inputArr, int numThreads);
template int fastHadamard3dArray_<float>(nb::ndarray<float, nb::shape<-1,-1,-1>,
                       nb::device::cpu, nb::c_contig> inputArr, int numThreads);




/*!
 * # fastHadamard2dArray_
 *
 * Performs an unnormalized Hadamard transform along the last
 * dimension of an input 2d array. The transform is performed
 * in place.
 *
 * ## Args:
 *
 * + `inputArr` A nanobind reference to a numpy array of shape
 * (N x C), where C must be a power of 2.
 * + `numThreads` The number of threads to use.
 */
template <typename T>
int fastHadamard2dArray_(nb::ndarray<T, nb::shape<-1,-1>,
                        nb::device::cpu, nb::c_contig> inputArr,
                        int numThreads){
    // Perform safety checks. Any exceptions thrown here are handed off to Python
    // by the Nanobind wrapper. We do not expect the user to see these because
    // the Python code will always ensure inputs are correct -- these are a failsafe
    // -- so we do not need to provide detailed exception messages here.
    int zDim0 = inputArr.shape(0);
    int zDim1 = inputArr.shape(1);
    T *inputPtr = static_cast<T*>(inputArr.data());

    if (inputArr.shape(0) == 0)
        throw std::runtime_error("no datapoints");
    if (inputArr.shape(1) < 2)
        throw std::runtime_error("last dim not power of 2 > 1");
    if ((zDim1 & (zDim1 - 1)) != 0)
        throw std::runtime_error("last dim not power of 2");

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
        threads[i] = std::thread(&ThreadTransformRows2D<T>, inputPtr,
                        startPosition, endPosition, zDim1);
    }

    for (auto& th : threads)
        th.join();
    return 0;
}
template int fastHadamard2dArray_<double>(nb::ndarray<double, nb::shape<-1,-1>,
                        nb::device::cpu, nb::c_contig> inputArr,
                        int numThreads);
template int fastHadamard2dArray_<float>(nb::ndarray<float, nb::shape<-1,-1>,
                        nb::device::cpu, nb::c_contig> inputArr,
                        int numThreads);






/*!
 * # SRHTBlockTransform_
 *
 * Performs the following operation along the last dimension
 * of the input array Z: H D1, where H is a normalized
 * Hadamard transform and D1 is a diagonal array.
 *
 * ## Args:
 *
 * + `inputArr` A nanobind reference to a numpy array of shape
 * (N x C), where C must be a power of 2.
 * + `radem` A diagonal array of shape (C) containing int8_t.
 * + `numThreads` The number of threads to use.
 */
template <typename T>
int SRHTBlockTransform(nb::ndarray<T, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> inputArr,
        nb::ndarray<int8_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> radem,
        int numThreads){

    // Perform safety checks. Any exceptions thrown here are handed off to Python
    // by the Nanobind wrapper. We do not expect the user to see these because
    // the Python code will always ensure inputs are correct -- these are a failsafe
    // -- so we do not need to provide detailed exception messages here.
    int zDim0 = inputArr.shape(0);
    int zDim1 = inputArr.shape(1);
    T *inputPtr = static_cast<T*>(inputArr.data());
    int8_t *rademPtr = static_cast<int8_t*>(radem.data());

    if (inputArr.shape(0) == 0)
        throw std::runtime_error("no datapoints");
    if (inputArr.shape(1) != radem.shape(0))
        throw std::runtime_error("incorrect array dims passed");
    if (inputArr.shape(1) < 2)
        throw std::runtime_error("last dim not power of 2 > 1");
    if ((zDim1 & (zDim1 - 1)) != 0)
        throw std::runtime_error("last dim not power of 2");

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
        threads[i] = std::thread(&ThreadSRHTRows2D<T>, inputPtr,
                rademPtr, zDim1, startPosition, endPosition);
    }

    for (auto& th : threads)
        th.join();
    return 0;
}
template int SRHTBlockTransform<double>(nb::ndarray<double, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> inputArr,
        nb::ndarray<int8_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> radem,
        int numThreads);
template int SRHTBlockTransform<float>(nb::ndarray<float, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> inputArr,
        nb::ndarray<int8_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> radem,
        int numThreads);






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
