Kernels for graphs
------------------------------------------------------

These are convolution kernels for graphs. To use one of these, when initializing the
model, set ``kernel_choice = 'kernel name'``, e.g.
``kernel_choice = "GraphRBF"``.


.. list-table:: Graph Kernels
   :align: center
   :header-rows: 1

   * - Kernel Name
     - Description
     - kernel_settings
   * - GraphRBF
     - | Compares graphs by averaging over
       | an RBF kernel applied pairwise to
       | all node representations
       | in the two graphs.
     - | "averaging": str
       | One of 'none', 'sqrt', 'full'. See
       | below.
       | "intercept":bool
   * - GraphMatern
     - | Compares graphs by averaging over
       | a Matern kernel applied pairwise to
       | all node representations
       | in the two graphs.
     - | "averaging": str
       | One of 'none', 'sqrt', 'full'. See
       | below.
       | "matern_nu":float
       | "intercept":bool
   * - GraphCauchy
     - | Compares graphs by averaging over
       | a Cauchy kernel applied pairwise to
       | all node representations
       | in the two graphs.
     - | "averaging": str
       | One of 'none', 'sqrt', 'full'. See
       | below.
       | "intercept":bool

Consider a graph where each node has an associated 
set of features. GraphRBF compares two graphs A and B by
taking each node in graph A and evaluating an RBF kernel across
that node vs each node in graph B. GraphMatern and GraphCauchy
do the same thing, except using Cauchy or Matern kernels.
A naive implementation would have quadratic scaling
in the size of the graph; in xGPR, remarkably, we are able to
implement these kernels with a "trick" that results in *linear
scaling* with graph size and number of datapoints for both kernels.

These convolution kernels can be slower than
fixed-vector input kernels, *especially* for large graphs,
because to limit memory use, the convolutions are performed
in batches (rather than all at once).

When using any of these kernels, you are required to supply ``sequence_lengths``
when building a dataset or doing inference. This is the number of nodes
in each graph. xGPR uses this information to mask out any zero-padding
you may have applied to make all the graphs the same size.

Note that all three offer averaging as an option. What this means
is as follows. The GraphRBF kernel computes the similarity of two
graphs with :math:`L_1` nodes in graph 1, :math:`L_2` nodes in graph 2 as:

.. math::

  k(x_1, x_2) = \sum_i^{L_1} \sum_j^{L_2} e^{\sigma ||x_1[i] - x_2[j]||^2}

Cauchy and Matern are the same except with Cauchy and Matern kernels substituted.

Notice this is actually performing :math:`L_1 * L_2` node comparisons
between the two, so the result will be larger when the graphs are larger. We can compensate
for this by dividing by :math:`L_1 * L_2`, which is ``full`` averaging, or dividing by :math:`\sqrt{L_1 * L_2}`, which is
``sqrt`` averaging. Averaging is helpful if the property you are trying to predict does not
depend on graph size. It is counterproductive if graph size actually *is* important.

Usually the validation set performance difference
between ``GraphMatern``, ``GraphCauchy`` and ``GraphRBF`` is 
small; if this is your primary concern, we recommend defaulting
to ``GraphRBF`` and experimenting with the others if desired to
see if some small further performance achievement can be obtained.
