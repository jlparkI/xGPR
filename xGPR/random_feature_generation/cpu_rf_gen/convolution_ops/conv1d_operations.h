#ifndef CONV1D_OPERATIONS_H
#define CONV1D_OPERATIONS_H


#include <stdint.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))


template <typename T>
int conv1dMaxpoolFeatureGen_(nb::ndarray<T, nb::shape<-1,-1,-1>, nb::device::cpu, nb::c_contig> inputArr,
        nb::ndarray<float, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> outputArr,
        nb::ndarray<int8_t, nb::shape<3, 1, -1>, nb::device::cpu, nb::c_contig> radem,
        nb::ndarray<T, nb::shape<-1>, nb::device::cpu, nb::c_contig> chiArr,
        nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> seqlengths,
        int convWidth, int numThreads);

template <typename T>
void *allInOneConvMaxpoolGen(T xdata[], int8_t *rademArray, T chiArr[],
        float *outputArray, int32_t *seqlengths, int dim1, int dim2,
        int numFreqs, int startRow, int endRow,
        int convWidth, int paddedBufferSize);

template <typename T>
void singleVectorMaxpoolPostProcess(const T xdata[],
        const T chiArr[], float *outputArray,
        int dim2, int numFreqs,
        int rowNumber, int repeatNum);

#endif
