What is xGPR
===============================================

xGPR is a library for fitting approximate Gaussian process regression
models to datasets ranging in size from hundreds to millions of datapoints,
and for performing other random-features based data analysis operations
(e.g. approximate kernel k-means and kernel PCA with random features).
It is designed to run on either CPU or GPU, to
model tabular data, sequence & time series data and graph data, and to
fit datasets too large to load into memory in a straightforward way.

Unlike other libraries for fitting GPs, xGPR is able to fit hundreds of
thousands of datapoints -- including graphs and sequences -- efficiently
while achieving performance competitive with deep learning for many tasks.


Limitations of xGPR
-------------------

xGPR can train on CPU, but will be substantially slower than if running on GPU.
We strongly encourage training on GPU for this reason. It is easy to train
on GPU then switch to CPU for inference if desired.
(If you are training on CPU, see :doc:`CPU training</cpu_training>`
for some helpful tricks.)

xGPR is not currently designed to perform classification -- indeed, it is hard
to do *fast and efficient* classification using GPs (slow and inefficient is
another matter!) We realize this rules out
problems of interest to many users. We are considering options for adding
classification capabilities in a future version.

xGPR uses a specific techinque to obtain scalable approximations to a
GP. This means that adding a novel kernel you have just cooked up to xGPR is
not straightforward. If your use-case is evaluating novel kernels xGPR may
not be your best choice.

xGPR is an approximation to a Gaussian process (which is what enables it to
achieve O(N) scaling). It is a good approximation, but no approximation
can consistently beat *exact* -- an *exact* GP with the same kernel will
perform a little better than xGPR. (We have found that in most cases the
benefit in switching to an exact GP is surpisingly modest!) 
If your dataset is sufficiently small that you can afford the cost of
an exact GP, you may prefer an exact GP instead.

Finally, Gaussian processes are powerful Bayesian models, but they come with
their own set of limitations. They are only as good as the kernel you select,
and like deep learning (but unlike gradient boosted trees), they are sensitive to scaling.
We think that GPs, gradient boosted trees and deep learning all have their "place" -- each
is most useful for certain kinds of problems. Our goal is not to provide a
library that solves *every* problem, but rather, the kinds of problems where
a GP is a better solution than a deep learning or tree-based model.
