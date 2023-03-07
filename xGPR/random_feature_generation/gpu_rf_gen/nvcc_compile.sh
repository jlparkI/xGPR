#!/bin/bash

nvcc --compiler-options '-fPIC'  \
    -c -o double_arrop_temp.o double_array_operations.cu
nvcc --compiler-options '-fPIC'  \
    -c -o float_arrop_temp.o float_array_operations.cu
nvcc --compiler-options '-fPIC'  \
    -c -o double_spec_temp.o rbf_ops/double_rbf_ops.cu
nvcc --compiler-options '-fPIC'  \
    -c -o float_spec_temp.o rbf_ops/float_rbf_ops.cu
nvcc --compiler-options '-fPIC'  \
    -c -o polyfht_temp.o polynomial_ops/poly_fht.cu
nvcc --compiler-options '-fPIC'  \
    -c -o conv_temp.o convolution_ops/convolution.cu
nvcc --compiler-options '-fPIC'  \
    -c -o rbf_conv_temp.o convolution_ops/rbf_convolution.cu

ar cru libarray_operations.a double_arrop_temp.o float_arrop_temp.o conv_temp.o polyfht_temp.o \
    float_spec_temp.o double_spec_temp.o rbf_conv_temp.o


ranlib libarray_operations.a

rm -f double_arrop_temp.o float_arrop_temp.o conv_temp.o polyfht_temp.o
rm -f float_spec_temp.o double_spec_temp.o rbf_conv_temp.o
