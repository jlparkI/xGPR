### Version 0.0.2.1

Pre-release for internal testing.

### Version 0.0.2.2

Pre-release for internal testing.

### Version 0.0.2.3

Pre-release for internal testing.

### Version 0.0.2.4

Initial release (first non-pre-release).

### Version 0.0.2.5

Updated release with additional options for
crude bayes tuning control.

### Version 0.0.2.6

Removed unneeded options for crude bayes tuning
control. Reset default cuda path which was set
incorrectly in earlier versions.

### Version 0.0.2.7

Rebuilt sdist using cuda-10.0. Removed earlier
versions to avoid any issues with default cuda path.

### Version 0.0.2.8

Fixed bugs preventing xGPR from running correctly
when no CUDA available.

### Version 0.0.2.9

Fixed an additional bug causing errors when running
with no CUDA available. CUDA-free operation now seems
to work properly.

### Version 0.0.3.0

Fixed bug causing errors on install in some Conda environments.

### Version 0.1.0.0

Added MiniARD and GraphMiniARD kernels. Extensive
refactoring of random feature generation code,
both for readability and speed, and of model
baseclass code, for readability. Removed deprecated
Conv1d kernel.

### Version 0.1.0.1

sdist for version 0.1.0.0 did not include required
.c / .cu files in error; corrected this error and
updated sdist

### Version 0.1.0.2

Version 0.1.0.1 limited the number of split points a
user could specify; this constraint has now been removed.

### Version 0.1.0.3

This version is able to calculate variance for Linear kernels,
has an improved variance calculation for all other kernels,
corrects some errors in the documentation, and the GraphConv1d
kernel has been renamed GraphRBF.

### Version 0.1.0.4

Removed references to CUDA PROFILER API, which may be problematic
in some environments.

### Version 0.1.0.5

visualization_tools was not included in previous build in error.

### Version 0.1.0.6

Removed deprecated kernels (e.g. GraphARD).

### Version 0.1.1.0

Major refactoring of C code for improved readability and efficiency.
Added GraphArcCos kernel which requires relatively little
hyperparameter tuning but usually does not perform quite so
well as GraphRBF.
