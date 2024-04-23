/*!
 * # simplex_rff_projections.cpp
 *
 * This module performs the simplex random feature projection
 * used for most RFF generation routines.
 */

#include <math.h>
#include "simplex_rff_projections.h"


/*!
 * # singleVectorSimplexProj
 *
 * Performs the simplex projection for a single vector.
 *
 * ## Args:
 *
 * + `cbuffer` Pointer to the first element of the 1d array. Size must
 * be a power of 2.
 * + `cbufferDim2` The size of cbuffer. Must be a power of 2.
 */
template <typename T>
void singleVectorSimplexProj(T cbuffer[], int cbufferDim2){
    T bufferSum = 0, scalar;
    T flBuffer = cbufferDim2;

    scalar = sqrt(flBuffer - 1.);

    for (int i=0; i < (cbufferDim2 - 1); i++)
        bufferSum += cbuffer[i];

    bufferSum /= scalar;
    cbuffer[cbufferDim2 - 1] = bufferSum;
    bufferSum *= ( (sqrt(flBuffer) + 1) / (flBuffer - 1) );
    scalar = sqrt( flBuffer / (flBuffer - 1) );

    for (int i=0; i < (cbufferDim2 - 1); i++)
        cbuffer[i] = (cbuffer[i] * scalar - bufferSum);
}
//Explicitly instantiate for external use.
template void singleVectorSimplexProj<double>(double cbuffer[], int cbufferDim2);
template void singleVectorSimplexProj<float>(float cbuffer[], int cbufferDim2);
