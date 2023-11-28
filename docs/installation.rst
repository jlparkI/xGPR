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

To run on GPU, you need Cupy. Running on GPU is highly recommended
for training.

xGPR is compiled from source. This makes installation a little
trickier unfortunately. At this time it is not provided on PyPi.
To install, visit the `github page <https://github.com/jlparkI/xGPR>`_
and click on Releases on the right hand side. Find the version number
for the latest release, then install it as follows:::

  pip install git+https://github.com/jlparkI/xGPR@0.0.1

but replace 0.0.1 with the latest version number. We'll provide
wheels on PyPi once xGPR development is complete.

When you run this, pip will try to compile the source code in
the distribution using *gcc*, *nvcc* and *g++*. xGPR will try to find CUDA
in the usual locations and if it cannot find it, will print a warning
if you are using *verbose*. This is the most common problem 
encountered when trying to compile xGPR. If you encounter this,
you can set CUDA_PATH as an environment variable to indicate
where CUDA is installed. You can do so by using the following
command:::

  export CUDA_PATH=/my/cuda/location

and pip will then use this cuda location when compiling the package.
