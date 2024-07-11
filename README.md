# xGPR

xGPR is a library for fitting approximate Gaussian process regression
models to datasets ranging in size from hundreds to millions of datapoints.
It uses an efficient implementation of the random features approximation
(aka random Fourier features). It is designed to run on either CPU or GPU
(GPU strongly preferred), to model tabular data, sequence & time series
data and graph data, and to fit datasets too large to load into memory
in a straightforward way.

Unlike exact Gaussian processes, which exhibit O(N^2) scaling
and are completely impractical for large datasets, xGPR can scale easily;
it is fairly straightforward to fit a few million datapoints
on a GPU. Notably, xGPR is able to do this while providing
accuracy competitive with deep learning models (*unlike* variational
GP approximations). Unlike other libraries for Gaussian processes,
which only provide kernels for fixed-vector data (tabular data),
xGPR provides powerful convolution kernels for variable-length time series,
sequences and graphs.

### What's new in v0.4.5
Starting with version 0.4.5, xGPR is available as a precompiled binary / wheel
for 64 bit Linux and as a source distribution for other platforms, so that
in most cases, installation should typically be as simple as:
```
pip install xGPR
```
See [the documentation](https://xgpr.readthedocs.io/en/latest/) for important
information about installation and requirements.


### Documentation

[The documentation](https://xgpr.readthedocs.io/en/latest/) covers a variety of use cases, including tabular data,
sequences and graphs, installation requirements and much more.

### Citations

If using xGPR for research intended for publication, please cite either:


Linear-Scaling Kernels for Protein Sequences and Small Molecules Outperform Deep Learning While Providing Uncertainty Quantitation and Improved Interpretability
Jonathan Parkinson and Wei Wang
Journal of Chemical Information and Modeling 2023 63 (15), 4589-4601
DOI: 10.1021/acs.jcim.3c00601 

or the preprint at:

Jonathan Parkinson, & Wei Wang. (2023). Linear Scaling Kernels for Protein Sequences and Small Molecules Outperform
Deep Learning while Providing Uncertainty Quantitation and Improved Interpretability
[https://arxiv.org/abs/2302.03294](https://arxiv.org/abs/2302.03294)
