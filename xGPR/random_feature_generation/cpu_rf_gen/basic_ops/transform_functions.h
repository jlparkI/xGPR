#ifndef TRANSFORM_FUNCTIONS_H
#define TRANSFORM_FUNCTIONS_H



template <typename T>
const char *fastHadamard3dArray_(T Z[], int zDim0, int zDim1, int zDim2,
                        int numThreads);

template <typename T>
const char *fastHadamard2dArray_(T Z[], int zDim0, int zDim1,
                        int numThreads);

template <typename T>
const char *SRHTBlockTransform_(T Z[], int8_t *radem,
            int zDim0, int zDim1, int numThreads);

template <typename T>
const char *SORFBlockTransform_(T Z[], int8_t *radem, int zDim0,
        int zDim1, int zDim2, int numThreads);

template <typename T>
void *ThreadSORFRows3D(T arrayStart[], int8_t* radem,
        int dim1, int dim2, int startPosition, int endPosition);

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
