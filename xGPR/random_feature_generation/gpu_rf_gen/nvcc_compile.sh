#!/bin/bash

nvcc --compiler-options '-fPIC'  \
    -c -o basic_arrop_temp.o basic_ops/basic_array_operations.cu
nvcc --compiler-options '-fPIC'  \
    -c -o rbf_spec_temp.o rbf_ops/rbf_ops.cu
nvcc --compiler-options '-fPIC'  \
    -c -o conv_temp.o convolution_ops/convolution.cu
nvcc --compiler-options '-fPIC'  \
    -c -o arccos_conv_temp.o convolution_ops/arccos_convolution.cu
nvcc --compiler-options '-fPIC'  \
    -c -o rbf_conv_temp.o convolution_ops/rbf_convolution.cu
nvcc --compiler-options '-fPIC'  \
    -c -o ard_conv_temp.o convolution_ops/ard_convolution.cu

ar cru libarray_operations.a basic_arrop_temp.o conv_temp.o \
    rbf_spec_temp.o rbf_conv_temp.o ard_conv_temp.o arccos_conv_temp.o


ranlib libarray_operations.a

rm -f basic_arrop_temp.o conv_temp.o ard_conv_temp.o
rm -f rbf_spec_temp.o rbf_conv_temp.o arccos_conv_temp.o
