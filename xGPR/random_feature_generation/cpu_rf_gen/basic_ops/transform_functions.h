#ifndef TRANSFORM_FUNCTIONS_H
#define TRANSFORM_FUNCTIONS_H


struct ThreadSORFFloatArrayArgs {
    int dim1, dim2;
    float *arrayStart;
    int startPosition, endPosition;
    int8_t *rademArray;
};


struct Thread3DFloatArrayArgs {
    int dim1, dim2;
    float *arrayStart;
    int startPosition, endPosition;
};

struct ThreadSORFDoubleArrayArgs {
    int dim1, dim2;
    double *arrayStart;
    int startPosition, endPosition;
    int8_t *rademArray;
};


struct Thread3DDoubleArrayArgs {
    int dim1, dim2;
    double *arrayStart;
    int startPosition, endPosition;
};


const char *fastHadamard3dFloatArray_(float *Z, int zDim0, int zDim1, int zDim2,
                        int numThreads);
const char *fastHadamard3dDoubleArray_(double *Z, int zDim0, int zDim1, int zDim2,
                        int numThreads);

const char *fastHadamard2dFloatArray_(float *Z, int zDim0, int zDim1,
                        int numThreads);
const char *fastHadamard2dDoubleArray_(double *Z, int zDim0, int zDim1,
                        int numThreads);

const char *SRHTFloatBlockTransform_(float *Z, int8_t *radem,
            int zDim0, int zDim1, int numThreads);
const char *SRHTDoubleBlockTransform_(double *Z, int8_t *radem,
            int zDim0, int zDim1, int numThreads);

const char *SORFFloatBlockTransform_(float *Z, int8_t *radem, int zDim0,
        int zDim1, int zDim2, int numThreads);
const char *SORFDoubleBlockTransform_(double *Z, int8_t *radem, int zDim0,
        int zDim1, int zDim2, int numThreads);

void *ThreadSORFFloatRows3D(void *rowArgs);
void *ThreadSORFDoubleRows3D(void *rowArgs);

void *ThreadSRHTFloatRows2D(void *rowArgs);
void *ThreadSRHTDoubleRows2D(void *rowArgs);

void *ThreadTransformRows3DFloat(void *rowArgs);
void *ThreadTransformRows3DDouble(void *rowArgs);

void *ThreadTransformRows2DFloat(void *rowArgs);
void *ThreadTransformRows2DDouble(void *rowArgs);

#endif
