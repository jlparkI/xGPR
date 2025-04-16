What is xGPR
===============================================

xGPR is a library for fitting approximate Gaussian process regression
models and kernel classification models to datasets ranging
in size from thousands to millions of datapoints.
It runs on either CPU or GPU and fits datasets too large to load
into memory in a straightforward way.


Limitations of xGPR
-------------------

xGPR can train on CPU, but will be substantially slower than if training on GPU.
We strongly encourage training / fitting on GPU for this reason. It is easy to train
on GPU then switch to CPU for inference if desired.

xGPR uses a specific techinque to obtain scalable approximations to a
GP. This means that adding a novel kernel you have just cooked up to xGPR is
not straightforward. If your use-case is evaluating novel kernels xGPR may
not be your best choice.

Finally, kernel machines can be powerful models, but they come with
their own set of limitations. They only work well if the kernel "makes sense" for
a given problem, and unlike gradient boosted trees, they are sensitive to scaling.
Our goal is not to provide a library that solves *every* problem, but rather,
the kinds of problems where an approximate kernel machine is a good solution.
