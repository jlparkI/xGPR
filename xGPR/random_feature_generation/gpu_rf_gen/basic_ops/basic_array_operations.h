#ifndef BASIC_CUDA_FHT_ARRAY_OPERATIONS_H
#define BASIC_CUDA_FHT_ARRAY_OPERATIONS_H

#include <stdint.h>


template <typename T>
void cudaHTransform(T cArray[],
		int dim0, int dim1, int dim2);


template <typename T>
const char *cudaSORF3d(T npArray[],  const int8_t *radem,
                int dim0, int dim1, int dim2);

template <typename T>
const char *cudaConvSORF3d(T cArray[], const int8_t *radem,
                int dim0, int dim1, int dim2,
                int startPosition, int numElements,
                int rademShape2, T normConstant);

template <typename T>
const char *cudaSRHT2d(T npArray[], 
                const int8_t *radem, int dim0,
                int dim1);


#endif
