Installation
================

Currently xGPR is only supported for Linux (support for
Windows will be added in a future release).

**Requirements**

* Python >= 3.9
* Numpy >= 1.9
* Scipy >= 1.7
* scikit-learn
* Cython

**Optional (but strongly recommended) dependencies**

* Cupy
* CUDA >= 10.0

To run on GPU, you need Cupy. Running on GPU accelerates xGPR
significantly and is HIGHLY recommended for training.

xGPR is compiled from source. The preferred approach is to run:::

  pip install xGPR

and pip will try to compile the source code in the distribution
using *gcc*, *nvcc* and *g++*. xGPR will try to find CUDA
in the usual locations and if it cannot find it, will print a warning
if you are using *verbose*. It will in this case prompt
you to set CUDA_PATH as an environment variable to indicate
where CUDA is installed. You can do so by using the following
command:::

  export CUDA_PATH=/my/cuda/location

and pip will then use this cuda location when compiling the package.

It is possible to clone the most recent version of the `github
repo <https://github.com/jlparki/xGPR>`_, but it is generally better
to use an existing release.
