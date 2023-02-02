#ifndef FLOAT_CUDA_ARRAY_OPERATIONS_H
#define FLOAT_CUDA_ARRAY_OPERATIONS_H

const char *floatCudaSORF3d(float *npArray, 
                    int8_t *radem, int dim0, int dim1,
                    int dim2);

const char *floatCudaSRHT2d(float *npArray, 
                    int8_t *radem, int dim0, int dim1);

void floatCudaHTransform3d(float *cArray,
		int dim0, int dim1, int dim2);
void floatCudaHTransform2d(float *cArray,
		int dim0, int dim1);

int getNumBlocksFloatTransform(int arrsize, int divisor);


#endif
