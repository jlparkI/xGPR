Kernels for graphs
------------------------------------------------------

These are convolution kernels for graphs, analogous to a graph
convolutional network. To use one of these, when initializing the
model, set ``kernel_choice = 'kernel name'``, e.g.
``kernel_choice = "GraphRBF"``.
See :doc:`Getting started</initialization_tutorial>`
for details.


.. list-table:: Sequence Kernels
   :align: center
   :header-rows: 1

   * - Kernel Name
     - Description
     - kernel_specific_params
   * - GraphRBF
     - | Compares graphs by averaging over
       | an RBF kernel applied pairwise to
       | all node representations
       | in the two graphs.
     - | "intercept": bool If True,
       | fit a y-intercept.
       | Defaults to True.
   * - GraphPoly
     - | Same as "GraphRBF", but applies
       | a polynomial kernel pairwise instead
       | of an RBF. Only two hyperparameters
       | that need to be tuned instead of 3
       | as for GraphRBF or FHTConv1d.
     - | "polydegree":int
       | "intercept": bool If True,
       | fit a y-intercept.
       | Defaults to True.
   * - GraphArcCosine
     - | Same as "GraphRBF", but applies
       | an arc-cosine kernel pairwise instead
       | of an RBF. Only two hyperparameters
       | that need to be tuned instead of 3
       | as for GraphRBF or FHTConv1d.
     - | "order":int, either 1 or 2. Determines
       | the order of the arc-cosine kernel.
       | "intercept": bool If True,
       | fit a y-intercept.
       | Defaults to True.

Consider a graph where each node has an associated 
set of features. GraphRBF compares two graphs A and B by
taking each node in graph A and evaluating an RBF kernel across
that node vs each node in graph B. GraphPoly does the same
thing, except it uses a polynomial kernel of the specified degree
to compare nodes. A naive implementation would have quadratic scaling
in the size of the graph; in xGPR, remarkably, we are able to
implement these kernels with a "trick" that results in *linear
scaling* with graph size and number of datapoints for both kernels.

Be aware that these convolution kernels are a little slower than
fixed-vector input kernels, *especially* for large graphs,
because to avoid using excessive
memory, the convolutions are performed in batches (rather
than all at once). As a compensating factor, they frequently
need fewer random features to achieve good performance.

Tuning hyperparameters for GraphArcCosine and GraphPoly is
usually quite straightforward since there are only two.
On the other hand, they are slightly slower than GraphRBF
for both fitting and inference.
