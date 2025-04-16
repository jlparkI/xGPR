#ifndef RBF_CONVOLUTION_H
#define RBF_CONVOLUTION_H

// C++ headers
#include <stdint.h>

// Library headers
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

// Project headers

namespace nb = nanobind;


namespace CPURBFConvolutionKernelCalculations {


#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
static constexpr int NO_CONVOLUTION_SCALING = 0;
static constexpr int SQRT_CONVOLUTION_SCALING = 1;
static constexpr int FULL_CONVOLUTION_SCALING = 2;



/// @brief Generates random features for RBF-based convolution kernels.
/// @param inputArr The input features that will be used to generate the RFs.
/// @param outputArr The array in which random features will be stored.
/// @param radem The array in which the diagonal Rademacher matrices are
/// stored.
/// @param chiArr The array in which the diagonal scaling matrix is stored.
/// @param seqlengths The array storing the length of each sequence in
/// inputArr.
/// @param convWidth The width of the convolution kernel.
/// @param scalingType The type of scaling to perform (i.e. how to normalize
/// for different sequence lengths).
template <typename T>
int convRBFFeatureGen_(nb::ndarray<T, nb::shape<-1,-1,-1>, nb::device::cpu, nb::c_contig> inputArr,
nb::ndarray<double, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> outputArr,
nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cpu, nb::c_contig> radem,
nb::ndarray<T, nb::shape<-1>, nb::device::cpu, nb::c_contig> chiArr,
nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> seqlengths,
int convWidth, int scalingType);



/// @brief Generates both random features and gradient for
/// RBF-based convolution kernels.
/// @param inputArr The input features that will be used to generate the RFs.
/// @param outputArr The array in which random features will be stored.
/// @param radem The array in which the diagonal Rademacher matrices are
/// stored.
/// @param chiArr The array in which the diagonal scaling matrix is stored.
/// @param seqlengths The array storing the length of each sequence in
/// inputArr.
/// @param gradArr The array in which the gradient will be stored.
/// @param convWidth The width of the convolution kernel.
/// @param sigma The lengthscale hyperparameter.
/// @param scalingType The type of scaling to perform (i.e. how to normalize
/// for different sequence lengths).
template <typename T>
int convRBFGrad_(nb::ndarray<T, nb::shape<-1,-1,-1>, nb::device::cpu, nb::c_contig> inputArr,
nb::ndarray<double, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> outputArr,
nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cpu, nb::c_contig> radem,
nb::ndarray<T, nb::shape<-1>, nb::device::cpu, nb::c_contig> chiArr,
nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> seqlengths,
nb::ndarray<double, nb::shape<-1,-1,1>, nb::device::cpu, nb::c_contig> gradArr,
double sigma, int convWidth, int scalingType);


}  // namespace CPURBFConvolutionKernelCalculations


#endif
