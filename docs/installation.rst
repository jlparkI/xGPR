Installation
================

Currently xGPR is only supported for Linux (support for
Windows will be added in a future release).

**Requirements**

* Python >= 3.9
* Numpy >= 1.9
* Scipy >= 1.7
* Cython

**Optional (but strongly recommended) dependencies**

* Cupy
* scikit-learn
* CUDA >= 10.0

To run on GPU, you need Cupy. Running on GPU accelerates xGPR
significantly and is HIGHLY recommended for training.

scikit-learn is needed if you want to use Bayesian optimization.
This can speed up hyperparameter tuning and is highly recommended.

xGPR is distributed as a source distribution. To install, run:::

  pip install xGPR

and pip will try to compile the source code in the distribution
using *gcc* and *nvcc*. Note that xGPR will try to find CUDA
in the usual locations and if it cannot find it, will prompt
you to set CUDA_PATH as an environment variable to indicate
where CUDA is installed. You can do so by using the following
command:::

  export CUDA_PATH=/my/cuda/location

If CUDA or cupy are not found, a warning will print at the end of
installation.
