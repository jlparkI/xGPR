Installation
================

Currently xGPR is only supported for Linux and is preferably
compiled using g++.

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

  pip install git+https://github.com/jlparkI/xGPR@0.4.0.1 --verbose

but replace 0.4.0.1 with the latest release number. We'll provide
wheels on PyPi once xGPR development is complete. We recommend
using the verbose flag as shown.

When you run this, pip will try to compile the source code in
the distribution using *gcc*, *nvcc* and *g++*. xGPR will try to find CUDA
in the usual locations -- it will, for example, use the output of
``which nvcc`` and if it cannot find it, will print a warning
if you are using *verbose*. This is the most common problem 
encountered when trying to compile xGPR. If you have used the
verbose flag as shown above, and this problem occurs, towards the end of
the installation, you'll see this message:::

  WARNING! the cuda installation was not found at the expected locations.
  xGPR has been installed for CPU use ONLY. If you want to run on GPU,
  please set the CUDA_PATH environment variable to indicate the location of your CUDA
  installation and reinstall.

If cuda compilation failed, you'll also encounter an error message like this
the first time you try to set an xGPR model to run on gpu.

If you encounter this, you can set CUDA_PATH as an environment variable to indicate
where CUDA is installed. You can do so by using the following
command:::

  export CUDA_PATH=/my/cuda/location

and pip will then use this cuda location when compiling the package.
These are the only currently known installation issues -- if you run
into an unexpected problem, please report it as an issue on github
and we'll fix it ASAP. For problems encountered while installing Cupy
which is also required for GPU operation, see the Cupy library docs.

Once this has run, installation is complete. Run:::

  pip show xGPR

to see the version number and other details.
