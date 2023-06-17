Kernels for sequence and time series data (non-static)
------------------------------------------------------

These kernels handle sequence and time series data,
similar to a 1d CNN with global average pooling.
To use one of these, when initializing the
model, set ``kernel_choice = 'kernel name'``, e.g.
``kernel_choice = "FHTConv1d"``. See
:doc:`Getting started</initialization_tutorial>`
for details.

*IMPORTANT NOTE*: In addition to these choices, you can use the
FastConv1d kernel for sequences, which is described under Static
Layers since it is implemented as a Static Layer rather than
as a kernel. FastConv1d compares sequences in a different way that
may in some cases be beneficial. See the graph below for some
comparisons.

.. list-table:: Sequence Kernels
   :align: center
   :header-rows: 1

   * - Kernel Name
     - Description
     - kernel_specific_params
   * - FHTConv1d
     - | Compares sequences by averaging over
       | an RBF kernel applied pairwise to
       | all subsequences of length "conv_width"
       | in the two sequences.
     - "conv_width":int


**Note:** The Conv1d kernel provided up through version 0.0.2.9
is deprecated, and is no longer available from 0.1.0.0 forward.
It provided the same functionality as FHTConv1d but was substantially
slower on CPU.

If we have a sequence (or time series) of length N and d = conv_width,
to measure the similarity of two sequences A and B, these kernels take all the
length d subsequences of A and for each length d subsequence in A,
evaluate an RBF kernel on it against all length d subsequences in B. The
net similarity is the average across all of these. If implemented as
described, of course, this kernel would be extremely inefficient. In xGPR,
however, we implement this kernel in such a way we can achieve *linear
scaling* in both number of datapoints and sequence length.

The chart below contrasts the performance of FHTConv1d, Conv1d
(a deprecated comparator used during testing) and
FastConv1d (a static layer kernel for sequences, see below) on some of the
FLIP benchmarks (a protein sequence benchmark) and the TAPE benchmarks
(similar), using one-hot encoded sequences as input and Spearman's r
as a measure of performance, with the same number of random features used
for all kernels. The error bar is across different random seeds. FHTConv1d
and Conv1d are similar as expected.

Be aware that these convolution kernels are slower than
fixed-vector input kernels, *especially* for long sequences,
because to avoid using excessive
memory, the convolutions are performed in batches (rather
than all at once). As a compensating factor, they frequently
need fewer random features to achieve good performance.

.. image:: images/conv_kernel_no_timing.png
