### Version 0.4.7
Minor updates to Dataset classes that simplify them so that
end users can create custom datasets and wrap e.g. csv files
or databases when setting up their training dataset.

### Version 0.4.6
Added the Conv1dTwoLayer kernel. Removed the experimental simplex
rffs feature which is of uncertain usefulness. Fixed a bug occuring
when non-contiguous arrays are passed to the static layer.

### Version 0.4.5
Updated all C++ code wrapping to nanobind; removed Cython routines;
switched to a CMake-based build workflow.

### Version 0.4.0.1
Fixed bug in Cuda generation of RFFs for > 1024 input features which caused fewer
RFFs than expected to be generated. Expanded unit tests to ensure > 1024 input
features is now tested.

### Version 0.4
Removed classification as an option (since the kernel discriminant is not an optimal
choice for this problem). Add classification back when / if a better classification
model can be found.

### Version 0.3.5.2
Simplified the FastConv1d API and updated the docs. FastConv1d now double checks
that the sequence lengths are int32. pre_prediction_checks now also ensures
sequence lengths are int32.

### Version 0.3.5
Achieved a close to 3x speedup for GPU random feature generation for
convolution kernels and significant further speedup for CPU convolution
kernels. Memory consumption now scales as O(1) with sequence length
for both CPU / GPU. Added the simplex random features (Reid et al. 2023)
modification as an option for most kernels, with the exception of
MiniARD and (of course) Linear.

### Version 0.3.2

Extensive modifications to wrapped C++ code for CPU convolution kernels
specifically, achieving a 25% speedup and a large reduction in memory
usage, which is now constant irrespective of sequence length. Fixed
bug in sequence length normalization.

### Version 0.3.1

Fixed bug in static layers module.

### Version 0.3

Extensive reworking of wrapped C++ / Cuda code for large improvements
in either memory usage or speed for both fixed vector and convolution
operations. Retired the poly and graph poly kernels, which are inefficient
and rarely useful. Introduced sequence length masking, so that any zero
padding we use is automatically masked if user so desires.

### Version 0.2.0.5

Fixed a bug in rf feature generation for RBF-based kernels for
situations where the number of input features is very large.

### Version 0.2.0.2

Altered the definition of averaging for conv kernels.

### Version 0.2.0.1

Fixed a bug in automatic preconditioner rank selection for
classification.

### Version 0.2

Greatly simplified the API and the hyperparameter tuning process.
Added automated preconditioner construction / rank selection.
Added classification using kernel discriminants. Improved
polynomial kernel. Updated graph averaging prefactor calculation.
Converted all kernels with two hyperparameters to one hyperparameter
and all kernels with three hyperparameters to two, by using the
automated beta selection routine. Removed kernelxPCA, which is
only intermittently useful.

### Version 0.1.3.2

Version for internal use; contains exact polynomial regression for
low-degree polynomials (a quadratic). Also experimenting with
alternative implementions of the polynomial kernel.

### Version 0.1.3.0

Added sequence / graph averaging to all convolution kernels as
an option. Added kernel PCA and clustering tools that do not require
a fitted model as input.

### Version 0.1.2.4

Fixed a bug involving variable-length sequence inputs to
FHT-conv1d kernels. Sped up the nan checking for dataset building.

### Version 0.1.2.3

Fixed a bug involving changing device after fitting from gpu
to cpu.

### Version 0.1.2.2

Updated dataset builder so that different batches with different
xdim[1] are now accepted when building a dataset. This obviates
the need to zero-pad data (although note that zero-padding can
still be used).

### Version 0.1.2.1

Added pyproject.toml file to force numpy preinstallation, thereby
avoiding issues encountered if xGPR is installed as part of a
requirements file with numpy not yet installed.

### Version 0.1.2.0

Added the option to either use / not use an intercept for
all kernels. Added the GraphArcCosine kernel to the docs.
Reconfigured all graph / sequence kernels to accept variable
length inputs when making predictions.
Updated data processing to improve memory consumption on
GPU for offline data during pretransform / hyperparameter
tuning operations. Updated scaling factor for RBF random
feature generation (this may change the value for the
beta hyperparameter selected by hyperparameter tuning
so that models tuned or fitted with a prior version of xGPR will
need to be retuned but should have no effect on performance).

### Version 0.1.1.6

Added the option to NOT normalize_y. Generally normalizing
y is beneficial, but there are some circumstances where
the user might not want to do so.

### Version 0.1.1.5

Fixed bug in variance / uncertainty calculation.

### Version 0.1.1.0

Major refactoring of C code for improved readability and efficiency.
Added GraphArcCos kernel which requires relatively little
hyperparameter tuning but usually does not perform quite so
well as GraphRBF.

### Version 0.1.0.6

Removed deprecated kernels (e.g. GraphARD).

### Version 0.1.0.5

visualization_tools was not included in previous build in error.

### Version 0.1.0.4

Removed references to CUDA PROFILER API, which may be problematic
in some environments.

### Version 0.1.0.3

This version is able to calculate variance for Linear kernels,
has an improved variance calculation for all other kernels,
corrects some errors in the documentation, and the GraphConv1d
kernel has been renamed GraphRBF.

### Version 0.1.0.2

Version 0.1.0.1 limited the number of split points a
user could specify; this constraint has now been removed.

### Version 0.1.0.1

sdist for version 0.1.0.0 did not include required
.c / .cu files in error; corrected this error and
updated sdist

### Version 0.1.0.0

Added MiniARD and GraphMiniARD kernels. Extensive
refactoring of random feature generation code,
both for readability and speed, and of model
baseclass code, for readability. Removed deprecated
Conv1d kernel.

### Version 0.0.3.0

Fixed bug causing errors on install in some Conda environments.

### Version 0.0.2.9

Fixed an additional bug causing errors when running
with no CUDA available. CUDA-free operation now seems
to work properly.

### Version 0.0.2.8

Fixed bugs preventing xGPR from running correctly
when no CUDA available.

### Version 0.0.2.7

Rebuilt sdist using cuda-10.0. Removed earlier
versions to avoid any issues with default cuda path.

### Version 0.0.2.6

Removed unneeded options for crude bayes tuning
control. Reset default cuda path which was set
incorrectly in earlier versions.

### Version 0.0.2.5

Updated release with additional options for
crude bayes tuning control.

### Version 0.0.2.4

Initial release (first non-pre-release).

### Version 0.0.2.3

Pre-release for internal testing.

### Version 0.0.2.2

Pre-release for internal testing.

### Version 0.0.2.1

Pre-release for internal testing.
