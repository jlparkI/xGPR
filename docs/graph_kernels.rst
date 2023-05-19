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
     -
   * - GraphPoly
     - | Same as "GraphRBF", but applies
       | a polynomial kernel pairwise instead
       | of an RBF. Only two hyperparameters
       | that need to be tuned instead of 3
       | as for GraphRBF or FHTConv1d.
     - | "polydegree":int
   * - GraphMiniARD (DEPRECATED -- removed in v0.1.0.6)
     - | Same as GraphRBF, but rather than having one
       | lengthscale shared between all features,
       | applies different lengthscales to different
       | groups of features. Much slower hyperparameter
       | tuning but can give better results for some problems.
     - | "split_points":list

**Note:** The GraphMiniARD kernel provided up through version 0.1.0.5
is deprecated, and is no longer available from 0.1.0.6 forward.
Hyperparameter tuning was relatively slow compared to other graph
kernels and we had not yet found a use case where it
provided a clear benefit.

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
