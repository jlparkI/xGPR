Kernels for sequence and time series data (non-static)
------------------------------------------------------

These kernels handle sequence and time series data,
similar to a 1d CNN with global average pooling.
To use one of these, when initializing the
model, set ``kernel_choice = 'kernel name'``, e.g.
``kernel_choice = "Conv1dRBF"``.

*IMPORTANT NOTE*: In addition to these choices, you can use the
FastConv1d kernel for sequences, which is described under Feature
Extractors since it is actually a feature extractor rather than
a typical kernel. FastConv1d compares sequences in a different way that
may in some cases be beneficial.

.. list-table:: Sequence Kernels
   :align: center
   :header-rows: 1

   * - Kernel Name
     - Description
     - kernel_settings
   * - Conv1dRBF
     - | Compares sequences by averaging over
       | an RBF kernel applied pairwise to
       | all subsequences of length "conv_width"
       | in the two sequences.
     - | "conv_width":int
       | "averaging": str One of 'none', 'sqrt',
       | 'full'. See below.
       | "intercept":bool
       | "simplex_rffs":bool . An experimental feature,
       | see below.
   * - Conv1dMatern
     - | Compares sequences by averaging over
       | a Matern kernel applied pairwise to
       | all subsequences of length "conv_width"
       | in the two sequences.
     - | "conv_width":int
       | "averaging": str One of 'none', 'sqrt',
       | 'full'. See below.
       | "intercept":bool
       | "matern_nu":float
       | "simplex_rffs":bool . An experimental feature,
       | see below.
   * - Conv1dCauchy
     - | Compares sequences by averaging over
       | a Cauchy kernel applied pairwise to
       | all subsequences of length "conv_width"
       | in the two sequences.
     - | "conv_width":int
       | "averaging": str One of 'none', 'sqrt',
       | 'full'. See below.
       | "intercept":bool
       | "simplex_rffs":bool . An experimental feature,
       | see below.


If we have a sequence (or time series) of length N and k = conv_width,
to measure the similarity of two sequences A and B, these kernels take all the
length k subsequences of A and for each length k subsequence in A,
evaluate an RBF or Cauchy or Matern kernel on it against all length d subsequences in B. The
net similarity is the sum across all of these. If implemented as
described, of course, this kernel would be extremely inefficient. In xGPR,
however, we implement this kernel in such a way we can achieve *linear
scaling* in both number of datapoints and sequence length.

Be aware that these convolution kernels are slower than
fixed-vector input kernels, *especially* for long sequences,
because to avoid using excessive
memory, the convolutions are performed in batches (rather
than all at once). As a compensating factor, they frequently
need fewer random features to achieve good performance.

When using any of these kernels, you are required to supply ``sequence_lengths``
when building a dataset or doing inference. This is the number of elements
in each sequence. xGPR uses this information to mask out any zero-padding
you may have applied to make all the sequences the same length.

Note that all Conv1d kernels offer averaging as an option. What this means
is as follows. The Conv1dRBF kernel computes the similarity of two
graphs for convolution width *k* with :math:`L_1` elements in sequence 1,
:math:`L_2` elements in sequence 2 as:

.. math::

  k(x_1, x_2) = \sum_i^{L_1 - k + 1} \sum_j^{L_2 - k + 1} e^{\sigma ||x_1[i:i+k] - x_2[j:j+k]||^2}

Cauchy and Matern are the same except with Cauchy and Matern kernels substituted.

Notice that if :math:`K_1 = L_1 - k + 1`, this is actually performing :math:`K_1 * K_2` k-mer comparisons
between the two, so the result will be larger when the sequences are longer. We can compensate
for this by dividing by :math:`K_1 * K_2`, which is ``full`` averaging, or dividing by :math:`\sqrt{K_1 * K_2}`, which is
``sqrt`` averaging. Averaging is helpful if the property you are trying to predict does not
depend on sequence length. It is counterproductive if sequence length actually *is* important.

Usually the validation set performance difference
between ``Conv1dMatern``, ``Conv1dCauchy`` and ``Conv1dRBF`` is 
small; if this is your primary concern, we recommend defaulting
to ``Conv1dRBF`` and experimenting with the others if desired to
see if some small further performance achievement can be obtained.

The simplex_rffs argument is an experimental feature which implements the
simplex rffs modification from Reid et al. 2023. This modification slightly
increases computational cost but (under some circumstances) slightly
decreases the number of RFFs required to achieve the same level of kernel
approximation. We haven't fully decided yet whether this modification is
worth keeping so it is experimental / not fully tested for now.
