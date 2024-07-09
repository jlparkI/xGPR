#ifndef TRANSFORM_FUNCTIONS_H
#define TRANSFORM_FUNCTIONS_H

#include <stdint.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;


template <typename T>
int fastHadamard3dArray_(nb::ndarray<T, nb::shape<-1,-1,-1>,
                       nb::device::cpu, nb::c_contig> inputArr, int numThreads);

template <typename T>
int fastHadamard2dArray_(nb::ndarray<T, nb::shape<-1,-1>,
                        nb::device::cpu, nb::c_contig> inputArr,
                        int numThreads);

template <typename T>
int SRHTBlockTransform(nb::ndarray<T, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> inputArr,
        nb::ndarray<int8_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> radem,
        int numThreads);

template <typename T>
void *ThreadSRHTRows2D(T arrayStart[], int8_t* rademArray,
        int dim1, int startPosition, int endPosition);

template <typename T>
void *ThreadTransformRows3D(T arrayStart[], int startPosition,
        int endPosition, int dim1, int dim2);

template <typename T>
void *ThreadTransformRows2D(T arrayStart[], int startPosition,
        int endPosition, int dim1);

#endif
