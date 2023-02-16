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

scikit-learn is needed if you want to use Bayesian optimization.
This can speed up hyperparameter tuning and is highly recommended.

xGPR must be compiled from source. The preferred approach is to run:::

  pip install xGPR --verbose

and pip will try to compile the source code in the distribution
using *gcc* and *nvcc*. You don't have to use *verbose*, but it's
preferable since xGPR will try to find CUDA
in the usual locations and if it cannot find it, will print a warning
if you are using *verbose*. It will in this case prompt
you to set CUDA_PATH as an environment variable to indicate
where CUDA is installed. You can do so by using the following
command:::

  export CUDA_PATH=/my/cuda/location

and pip will then use this cuda location when compiling the package.

**Known issue**: Occasionally we have encountered issues when trying
to install into a Conda environment, since in this case the CUDA path
may be set to a Conda installation of cuda that does not always contain
the necessary libraries for compilation. This problem can be avoided by
setting ``export CUDA_PATH`` as above. We prefer to install into a virtual
environment rather than a Conda environment where possible since it avoids
this whole issue.

An alternative way to install is to retrieve the most recent
release from `the github releases page <https://github.com/jlparkI/xGPR/releases>`_ .
Download and extract the zipped source file, then inside the unzipped
source directory, run:::

  python setup.py install


Finally, it is possible to clone the most recent version of the `github
repo <https://github.com/jlparki/xGPR>`_, but it is generally better
to use an existing release.
