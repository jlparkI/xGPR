/* Copyright (C) 2025 Jonathan Parkinson
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


template <typename T>
int fastHadamard3dArray_(nb::ndarray<T, nb::shape<-1, -1, -1>,
    nb::device::cpu, nb::c_contig> input_arr) {

    // Perform safety checks. Any exceptions thrown here are handed
    // off to Python by the Nanobind wrapper. We do not expect the user
    // to see these because the Python code will always ensure inputs are
    // correct -- these are a failsafe -- so we do not need to provide
    // detailed exception messages here.
    int xDim0 = input_arr.shape(0);
    int xDim1 = input_arr.shape(1);
    int xDim2 = input_arr.shape(2);
    T *inputPtr = static_cast<T*>(input_arr.data());

    if (input_arr.shape(0) == 0)
        throw std::runtime_error("no datapoints");
    if (input_arr.shape(2) < 2)
        throw std::runtime_error("last dim not power of 2 > 1");
    if ((xDim2 & (xDim2 - 1)) != 0)
        throw std::runtime_error("last dim not power of 2");


    CPUHadamardTransformOps::transformRows<T>(inputPtr, 0,
                    xDim0, xDim1, xDim2);
    return 0;
}
template int fastHadamard3dArray_<double>(
nb::ndarray<double, nb::shape<-1, -1, -1>,
nb::device::cpu, nb::c_contig> input_arr);

template int fastHadamard3dArray_<float>(
nb::ndarray<float, nb::shape<-1, -1, -1>,
nb::device::cpu, nb::c_contig> input_arr);




template <typename T>
int fastHadamard2dArray_(nb::ndarray<T, nb::shape<-1, -1>,
                        nb::device::cpu, nb::c_contig> input_arr) {
    // Perform safety checks. Any exceptions thrown here are handed
    // off to Python by the Nanobind wrapper. We do not expect the user
    // to see these because the Python code will always ensure inputs are
    // correct -- these are a failsafe -- so we do not need to provide
    // detailed exception messages here.
    int xDim0 = input_arr.shape(0);
    int xDim1 = input_arr.shape(1);
    T *inputPtr = static_cast<T*>(input_arr.data());

    if (input_arr.shape(0) == 0)
        throw std::runtime_error("no datapoints");
    if (input_arr.shape(1) < 2)
        throw std::runtime_error("last dim not power of 2 > 1");
    if ((xDim1 & (xDim1 - 1)) != 0)
        throw std::runtime_error("last dim not power of 2");

    CPUHadamardTransformOps::transformRows<T>(inputPtr, 0,
                    xDim0, 1, xDim1);
    return 0;
}
template int fastHadamard2dArray_<double>(
nb::ndarray<double, nb::shape<-1, -1>,
nb::device::cpu, nb::c_contig> input_arr);

template int fastHadamard2dArray_<float>(
nb::ndarray<float, nb::shape<-1, -1>,
nb::device::cpu, nb::c_contig> input_arr);






template <typename T>
int SRHTBlockTransform(nb::ndarray<T, nb::shape<-1, -1>, nb::device::cpu, nb::c_contig> input_arr,
nb::ndarray<int8_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> radem) {
    // Perform safety checks. Any exceptions thrown here are handed
    // off to Python by the Nanobind wrapper. We do not expect the user
    // to see these because the Python code will always ensure inputs are
    // correct -- these are a failsafe -- so we do not need to provide
    // detailed exception messages here.
    int xDim0 = input_arr.shape(0);
    int xDim1 = input_arr.shape(1);
    T *inputPtr = static_cast<T*>(input_arr.data());
    int8_t *rademPtr = static_cast<int8_t*>(radem.data());

    if (input_arr.shape(0) == 0)
        throw std::runtime_error("no datapoints");
    if (input_arr.shape(1) != radem.shape(0))
        throw std::runtime_error("incorrect array dims passed");
    if (input_arr.shape(1) < 2)
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
nb::ndarray<double, nb::shape<-1, -1>, nb::device::cpu, nb::c_contig> input_arr,
nb::ndarray<int8_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> radem);

template int SRHTBlockTransform<float>(
nb::ndarray<float, nb::shape<-1, -1>, nb::device::cpu, nb::c_contig> input_arr,
nb::ndarray<int8_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> radem);

}  // namespace CPUHadamardTransformBasicCalculations
