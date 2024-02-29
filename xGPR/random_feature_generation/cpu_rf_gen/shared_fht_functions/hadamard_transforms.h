#ifndef HADAMARD_TRANSFORM_OPERATIONS_H
#define HADAMARD_TRANSFORM_OPERATIONS_H


template <typename T>
void transformRows(T __restrict xArray[], int startRow, int endRow,
                    int dim1, int dim2);

template <typename T>
void smallBlockTransform(T xArray[], int startRow, int endRow,
                    int dim1, int dim2);

template <typename T>
void generalTransform(T xArray[], int startRow, int endRow,
                    int dim1, int dim2);

#endif
