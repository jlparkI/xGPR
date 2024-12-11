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
a typical kernel. FastConv1d is equivalent to the ``Conv1dTwoLayer``
kernel described below but sometimes using the feature extractor
in preference to the kernel shown here can be useful.

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
   * - Conv1dCauchy
     - | Compares sequences by averaging over
       | a Cauchy kernel applied pairwise to
       | all subsequences of length "conv_width"
       | in the two sequences.
     - | "conv_width":int
       | "averaging": str One of 'none', 'sqrt',
       | 'full'. See below.
       | "intercept":bool
   * - Conv1dTwoLayer
     - | Compares sequences by performing random-weight
       | convolutions over the input, applying ReLU
       | activation with global maxpooling, then
       | supplying the resulting features as input
       | to an RBF kernel layer.
     - | "init_rffs": int The number of random
       | filter convolutions to perform.
       | "intercept": bool
       | "conv_width": The width of the random filters.


The ``Conv1dTwoLayer`` kernel is analogous to a three-layer convolutional
neural network; it applies a set of random filters to the input, applies
ReLU activation and global maxpooling, then
uses the resulting features as input to an RBF kernel. You can control
the number of random filters using the "init_rffs" option. A larger
value for "init_rffs" will make the model slower but improve accuracy
(albeit with diminishing returns).

If we have a sequence (or time series) of length N and k = conv_width,
to measure the similarity of two sequences A and B, the ``Conv1dMatern``,
``Conv1dRBF`` and ``Conv1dCauchy`` take all the
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

Note that all except the ``Conv1dTwoLayer`` kernel offer averaging as an
option. What this means is as follows. The Conv1dRBF kernel computes the similarity of two
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
``Conv1dTwoLayer`` by contrast can sometimes perform significantly
better (or worse) than these other options.
