#ifndef SPEC_CPU_RBF_OPS_H
#define SPEC_CPU_RBF_OPS_H

#include <stdint.h>
#include "nanobind/nanobind.h"
#include <nanobind/ndarray.h>

namespace nb = nanobind;

template <typename T>
int rbfFeatureGen_(nb::ndarray<T, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> inputArr,
        nb::ndarray<double, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> outputArr,
        nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cpu, nb::c_contig> radem,
        nb::ndarray<T, nb::shape<-1>, nb::device::cpu, nb::c_contig> chiArr,
        int numThreads, bool fitIntercept);

template <typename T>
int rbfGrad_(nb::ndarray<T, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> inputArr,
        nb::ndarray<double, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> outputArr,
        nb::ndarray<double, nb::shape<-1,-1,1>, nb::device::cpu, nb::c_contig> gradArr,
        nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cpu, nb::c_contig> radem,
        nb::ndarray<T, nb::shape<-1>, nb::device::cpu, nb::c_contig> chiArr,
        float sigma, int numThreads, bool fitIntercept);


template <typename T>
void *allInOneRBFGen(T xdata[], int8_t *rademArray, T chiArr[],
        double *outputArray, int dim1, int numFreqs, int rademShape2,
        int startRow, int endRow, int paddedBufferSize,
        double scalingTerm);


template <typename T>
void *allInOneRBFGrad(T xdata[], int8_t *rademArray, T chiArr[],
        double *outputArray, double *gradientArray,
        int dim1, int numFreqs, int rademShape2, int startRow,
        int endRow, int paddedBufferSize,
        double scalingTerm, T sigma);

#endif
