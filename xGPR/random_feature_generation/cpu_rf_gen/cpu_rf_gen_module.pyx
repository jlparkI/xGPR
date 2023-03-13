"""This 'main' Cython extension combines all the other
CPU Cython extensions so that all of them are built as
a single .so. This is slightly clunky, but at this time
there does not appear to be a better way to merge
multiple .pyx files to a single extension."""

cimport cython
include "cpu_basic_operations.pyx"
include "cpu_convolution_double.pyx"
include "cpu_convolution_float.pyx"
include "cpu_polynomial.pyx"
include "cpu_rbf_operations.pyx"
