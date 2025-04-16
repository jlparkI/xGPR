/* Copyright (C) 2025 Jonathan Parkinson
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

// C++ headers
#include <vector>
#include <stdexcept>

// Library headers

// Project headers
#include "transform_functions.h"
#include "../shared_fht_functions/hadamard_transforms.h"
#include "../shared_fht_functions/shared_rfgen_ops.h"

namespace nb = nanobind;


namespace CPUHadamardTransformBasicCalculations {


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
int fastHadamard3dArray_(nb::ndarray<T, nb::shape<-1, -1, -1>,
    nb::device::cpu, nb::c_contig> inputArr) {

    // Perform safety checks. Any exceptions thrown here are handed
    // off to Python by the Nanobind wrapper. We do not expect the user
    // to see these because the Python code will always ensure inputs are
    // correct -- these are a failsafe -- so we do not need to provide
    // detailed exception messages here.
    int xDim0 = inputArr.shape(0);
    int xDim1 = inputArr.shape(1);
    int xDim2 = inputArr.shape(2);
    T *inputPtr = static_cast<T*>(inputArr.data());

    if (inputArr.shape(0) == 0)
        throw std::runtime_error("no datapoints");
    if (inputArr.shape(2) < 2)
        throw std::runtime_error("last dim not power of 2 > 1");
    if ((xDim2 & (xDim2 - 1)) != 0)
        throw std::runtime_error("last dim not power of 2");


    CPUHadamardTransformOps::transformRows<T>(inputPtr, 0,
                    xDim0, xDim1, xDim2);
    return 0;
}
template int fastHadamard3dArray_<double>(
nb::ndarray<double, nb::shape<-1, -1, -1>,
nb::device::cpu, nb::c_contig> inputArr);

template int fastHadamard3dArray_<float>(
nb::ndarray<float, nb::shape<-1, -1, -1>,
nb::device::cpu, nb::c_contig> inputArr);




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
int fastHadamard2dArray_(nb::ndarray<T, nb::shape<-1, -1>,
                        nb::device::cpu, nb::c_contig> inputArr) {
    // Perform safety checks. Any exceptions thrown here are handed
    // off to Python by the Nanobind wrapper. We do not expect the user
    // to see these because the Python code will always ensure inputs are
    // correct -- these are a failsafe -- so we do not need to provide
    // detailed exception messages here.
    int xDim0 = inputArr.shape(0);
    int xDim1 = inputArr.shape(1);
    T *inputPtr = static_cast<T*>(inputArr.data());

    if (inputArr.shape(0) == 0)
        throw std::runtime_error("no datapoints");
    if (inputArr.shape(1) < 2)
        throw std::runtime_error("last dim not power of 2 > 1");
    if ((xDim1 & (xDim1 - 1)) != 0)
        throw std::runtime_error("last dim not power of 2");

    CPUHadamardTransformOps::transformRows<T>(inputPtr, 0,
                    xDim0, 1, xDim1);
    return 0;
}
template int fastHadamard2dArray_<double>(
nb::ndarray<double, nb::shape<-1, -1>,
nb::device::cpu, nb::c_contig> inputArr);

template int fastHadamard2dArray_<float>(
nb::ndarray<float, nb::shape<-1, -1>,
nb::device::cpu, nb::c_contig> inputArr);






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
int SRHTBlockTransform(nb::ndarray<T, nb::shape<-1, -1>, nb::device::cpu, nb::c_contig> inputArr,
        nb::ndarray<int8_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> radem) {

    // Perform safety checks. Any exceptions thrown here are handed
    // off to Python by the Nanobind wrapper. We do not expect the user
    // to see these because the Python code will always ensure inputs are
    // correct -- these are a failsafe -- so we do not need to provide
    // detailed exception messages here.
    int xDim0 = inputArr.shape(0);
    int xDim1 = inputArr.shape(1);
    T *inputPtr = static_cast<T*>(inputArr.data());
    int8_t *rademPtr = static_cast<int8_t*>(radem.data());

    if (inputArr.shape(0) == 0)
        throw std::runtime_error("no datapoints");
    if (inputArr.shape(1) != radem.shape(0))
        throw std::runtime_error("incorrect array dims passed");
    if (inputArr.shape(1) < 2)
        throw std::runtime_error("last dim not power of 2 > 1");
    if ((xDim1 & (xDim1 - 1)) != 0)
        throw std::runtime_error("last dim not power of 2");

    SharedCPURandomFeatureOps::multiplyByDiagonalRademacherMat2D<T>(
            inputPtr, rademPtr, xDim1, 0, xDim0);
    CPUHadamardTransformOps::transformRows<T>(inputPtr, 0,
                    xDim0, 1, xDim1);
    return 0;
}
template int SRHTBlockTransform<double>(
nb::ndarray<double, nb::shape<-1, -1>, nb::device::cpu, nb::c_contig> inputArr,
nb::ndarray<int8_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> radem);

template int SRHTBlockTransform<float>(
nb::ndarray<float, nb::shape<-1, -1>, nb::device::cpu, nb::c_contig> inputArr,
nb::ndarray<int8_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> radem);






/*!
 * # ThreadTransformRows2D
 *
 * Performs the fast Hadamard transform for one thread for a chunk of
 * the input array Z from startRow through endRow (each thread
 * works on its own group of rows).
 */
template <typename T>
void *ThreadTransformRows2D(T arrayStart[], int startPosition,
        int endPosition, int dim1) {
    CPUHadamardTransformOps::transformRows<T>(arrayStart, startPosition,
                    endPosition, 1, dim1);
    return NULL;
}



}  // namespace CPUHadamardTransformBasicCalculations
