#ifndef RBF_CONVOLUTION_H
#define RBF_CONVOLUTION_H

#include <stdint.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;


#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define NO_CONVOLUTION_SCALING 0
#define SQRT_CONVOLUTION_SCALING 1
#define FULL_CONVOLUTION_SCALING 2



template <typename T>
int convRBFFeatureGen_(nb::ndarray<T, nb::shape<-1,-1,-1>, nb::device::cpu, nb::c_contig> inputArr,
        nb::ndarray<double, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> outputArr,
        nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cpu, nb::c_contig> radem,
        nb::ndarray<T, nb::shape<-1>, nb::device::cpu, nb::c_contig> chiArr,
        nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> seqlengths,
        int convWidth, int scalingType, int numThreads);

template <typename T>
int convRBFGrad_(nb::ndarray<T, nb::shape<-1,-1,-1>, nb::device::cpu, nb::c_contig> inputArr,
        nb::ndarray<double, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> outputArr,
        nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cpu, nb::c_contig> radem,
        nb::ndarray<T, nb::shape<-1>, nb::device::cpu, nb::c_contig> chiArr,
        nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> seqlengths,
        nb::ndarray<double, nb::shape<-1,-1,1>, nb::device::cpu, nb::c_contig> gradArr,
        double sigma, int convWidth, int scalingType, int numThreads);

template <typename T>
void *allInOneConvRBFGen(T xdata[], int8_t *rademArray, T chiArr[],
        double *outputArray, int32_t *seqlengths, int dim1, int dim2,
        int numFreqs, int rademShape2, int startRow, int endRow,
        int convWidth, int paddedBufferSize,
        double scalingTerm, int scalingType);

template <typename T>
void *allInOneConvRBFGrad(T xdata[], int8_t *rademArray, T chiArr[],
        double *outputArray, int32_t *seqlengths, double *gradientArray,
        int dim1, int dim2, int numFreqs, int rademShape2, int startRow,
        int endRow, int convWidth, int paddedBufferSize,
        double scalingTerm, int scalingType, T sigma);

#endif
