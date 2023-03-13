"""This 'main' Cython extension combines all the other
Cuda Cython extensions so that all of them are built as
a single .so. This is slightly clunky, but at this time
there does not appear to be a better way to merge
multiple .pyx files to a single extension."""

cimport cython
include "cuda_basic_operations.pyx"
include "cuda_convolution_double.pyx"
include "cuda_convolution_float.pyx"
include "cuda_polynomial.pyx"
include "cuda_rbf_operations.pyx"
