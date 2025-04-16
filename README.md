# xGPR

xGPR is a library for fitting approximate Gaussian process regression
models and approximate kernel classification models to datasets ranging
in size from thousands to millions of datapoints. It can also be used
for efficient approximate kernel k-means and approximate kernel PCA.

[The docs](https://xgpr.readthedocs.io/en/latest/) provide a number of
examples for how to use xGPR for fitting protein sequences, small
molecule structures, and tabular data for regression (classification
is also available in v0.4.8).

xGPR uses a fast Hadamard transform-based implementation of the random features
approximation (aka random Fourier features). It is designed to run on either CPU
or GPU (GPU is better for training, either is fine for inference),
to model tabular data, sequence & time series data and graph data, and to
fit datasets too large to load into memory in a straightforward way.

Unlike exact Gaussian processes, which exhibit O(N^2) scaling
and are impractical for large datasets, xGPR can scale easily;
it is straightforward to quickly fit a few million datapoints
on a GPU. The approximation we use provides improved accuracy
compared with variational or sparse GP approximations.
Unlike other libraries for Gaussian processes,
which only provide kernels for fixed-vector data,
xGPR provides convolution kernels for variable-length time series,
sequences and graphs.

### What's new in v0.4.8
An approximate kernel classifier is now included. Unlike the xGPRegression
object, this does not currently however compute marginal likelihood, so
to tune hyperparameters for this you will have to evaluate performance
on a validation set. We hope to implement approximate marginal likelihood
calculations for this soon.

You can now build custom Datasets (similar to the Dataloader in PyTorch)
so that you can use any kind of data (SQLite db, HDF5 etc.) as input
when training with minor tweaks.

In most cases, installation should typically be as simple as:
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

The preprint is available at:

Jonathan Parkinson, & Wei Wang. (2023). Linear Scaling Kernels for Protein Sequences and Small Molecules Outperform
Deep Learning while Providing Uncertainty Quantitation and Improved Interpretability
[https://arxiv.org/abs/2302.03294](https://arxiv.org/abs/2302.03294)
