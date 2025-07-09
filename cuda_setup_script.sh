#!/bin/bash

rm -rf build

yum install -y wget

# Install the cuda11 toolkit.
wget https://developer.download.nvidia.com/compute/cuda/11.0.1/local_installers/cuda-repo-rhel7-11-0-local-11.0.1_450.36.06-1.x86_64.rpm
rpm -i cuda-repo-rhel7-11-0-local-11.0.1_450.36.06-1.x86_64.rpm
yum clean all
yum install -y cuda
rm -rf cuda-repo-rhel7-11-0-local-11.0.1_450.36.06-1.x86_64.rpm

# Add the newly installed cuda toolkit to path.
export PATH=/usr/local/cuda-11.0/bin/:$PATH
export PATH=/usr/local/cuda-11.0/:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64:$LD_LIBRARY_PATH

# For Cuda 11, we have to use gcc / g++ 9 or earlier. This is compatible with
# C++17 (but the gcc / g++ version required for cuda 10 may not be, which is
# why we do not support cuda 10).
yum install -y devtoolset-9
scl enable devtoolset-9 bash
