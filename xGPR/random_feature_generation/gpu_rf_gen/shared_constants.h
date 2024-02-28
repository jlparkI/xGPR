#ifndef CUDA_SHARED_CONSTANTS_ACROSS_OPS_H
#define CUDA_SHARED_CONSTANTS_ACROSS_OPS_H

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

#define DEFAULT_THREADS_PER_BLOCK 256
#define DEFAULT_THREADS_PER_BLREDUCE 32

#define MAX_BASE_LEVEL_TRANSFORM 1024
#define MAX_SINGLE_STAGE_TRANSFORM 1024


#endif
