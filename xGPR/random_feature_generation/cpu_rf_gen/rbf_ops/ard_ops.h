#ifndef SPEC_CPU_ARD_OPS_H
#define SPEC_CPU_ARD_OPS_H

#include <stdint.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;


template <typename T>
int ardGrad_(nb::ndarray<T, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> inputArr,
        nb::ndarray<double, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> outputArr,
        nb::ndarray<T, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> precompWeights,
        nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> sigmaMap,
        nb::ndarray<double, nb::shape<-1>, nb::device::cpu, nb::c_contig> sigmaVals,
        nb::ndarray<double, nb::shape<-1,-1,-1>, nb::device::cpu, nb::c_contig> gradArr,
        int numThreads, bool fitIntercept);



template <typename T>
void *ThreadARDGrad(T inputX[], double *randomFeatures,
        T precompWeights[], int32_t *sigmaMap,
        double *sigmaVals, double *gradient,
        int startRow, int endRow,
        int dim1, int numLengthscales,
        int numFreqs, double rbfNormConstant);

#endif
