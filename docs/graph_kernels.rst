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
   * - GraphMiniARD
     - | Same as GraphRBF, but rather than having one
       | lengthscale shared between all features,
       | applies different lengthscales to different
       | groups of features. Much slower hyperparameter
       | tuning but can give better results for some problems.
     - | "split_points":list


Consider a graph where each node has an associated 
set of features. GraphRBF compares two graphs A and B by
taking each node in graph A and evaluating an RBF kernel across
that node vs each node in graph B. GraphPoly does the same
thing, except it uses a polynomial kernel of the specified degree
to compare nodes. A naive implementation would have quadratic scaling
in the size of the graph; in xGPR, remarkably, we are able to
implement these kernels with a "trick" that results in *linear
scaling* with graph size and number of datapoints for both kernels.

The ``GraphMiniARD`` is a GraphRBF kernel that assigns a different lengthscale
to different kinds of features for each node. You might have data, for example,
where some features for each node describe that node, and some other features
for that node describe its neighbors. If
so, you could use GraphMiniARD and "learn" a different lengthscale for
each type of feature. For this kernel, supply a list under
``kernel_specific_params`` when creating a model, e.g.:::

  my_model = xGPRegression(training_rffs = 2048, fitting_rffs = 8192,
                        variance_rffs = 512, kernel_choice = "GraphMiniARD",
                        device = "gpu", kernel_specific_params =
                        {"kernel_specific_params":[21,36])

The features in between two split points all share a lengthscale. In this
case, for example, features from 0:21 in the input for each node would share one
lengthscale, features from 21:36 would share another, and features from
36: would share another lengthscale (0 and len(feature_vector) are automatically
added to the beginning and end of split_points). This technique can be
very powerful but also does make tuning more complicated and more time-consuming,
especially if the number of lengthscales is very large, so use judiciously.
The lengthscales learned by ``MiniARD`` during tuning can be used as crude
measures of relative importance (larger = more important group of features).

Be aware that these convolution kernels are a little slower than
fixed-vector input kernels, *especially* for large graphs,
because to avoid using excessive
memory, the convolutions are performed in batches (rather
than all at once). As a compensating factor, they frequently
need fewer random features to achieve good performance.
