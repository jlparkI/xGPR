#ifndef HADAMARD_TRANSFORM_OPERATIONS_H
#define HADAMARD_TRANSFORM_OPERATIONS_H


template <typename T>
void transformRows3D(T xArray[], int startRow, int endRow,
                    int dim1, int dim2);

template <typename T>
void transformRows2D(T xArray[], int startRow, int endRow,
                    int dim1);
#endif
