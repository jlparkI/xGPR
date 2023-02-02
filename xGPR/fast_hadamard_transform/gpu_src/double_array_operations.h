#ifndef DOUBLE_CUDA_ARRAY_OPERATIONS_H
#define DOUBLE_CUDA_ARRAY_OPERATIONS_H

const char *doubleCudaSORF3d(double *npArray, 
                    int8_t *radem, int dim0, int dim1,
                    int dim2);

const char *doubleCudaSRHT2d(double *npArray, 
                    int8_t *radem, int dim0, int dim1);

void doubleCudaHTransform3d(double *cArray,
		int dim0, int dim1, int dim2);
void doubleCudaHTransform2d(double *cArray,
		int dim0, int dim1);

int getNumBlocksDoubleTransform(int arrsize, int divisor);


#endif
