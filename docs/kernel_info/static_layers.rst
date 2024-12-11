Feature extractors for sequences and graphs
---------------------------------------------

The ``Conv1dTwoLayer`` kernel in xGPR can also be set up as a feature
extractor or "static layer". A ``static_layer`` is a feature extractor
applied to the input data that
converts it to a form that an xGPR model with a ``Linear``, ``RBF`` or
``Matern`` kernel can use as input. These are called 
``static_layers`` because they do not contain any tunable 
hyperparameters, so they can be applied to all of the training 
data *once* before the start of training. When making 
predictions for new datapoints, the ``static_layer`` must of 
course be applied to the new datapoints as well. To learn
how to do use static layers, see the Advanced tutorials.

.. list-table:: static layer kernels
   :header-rows: 1

   * - Kernel Name
     - Description
   * - FastConv1d
     - | Performs random-feature convolutions across
       | input sequences to build a sequence profile
       | for each datapoint, then uses an RBF (or
       | potentially Linear or Matern) kernel to
       | compare the sequence profile of each
       | datapoint.

``FastConv1d`` is a way to compare
sequences and time series that is analogous to a three-layer
convolutional neural network. In the first (static) layer, random
feature convolutions are applied followed by global max pooling;
this essentially measures what is the "best match" for a given
random filter in the sequence of interest and thereby creates a
"sequence profile". In the second and third layer, an ``RBF``
(or potentially other kernels, although we've only really used
``RBF``) kernel compares the sequence profile of training datapoints.

If using these kernels, choose an ``RBF`` kernel when initializing
the xGPRegression model (for ``FastConv1d``). (You *could* also use
Linear or Matern or some other in principle, but we've never found that to be
terribly useful.)

Using the output of a ``FastConv1d`` feature extractor as input to an RBF
kernel is equivalent to using the ``Conv1dTwoLayer`` kernel. Why would you
prefer the static layer? There are some situations where it can be more
convenient. For example, if you are supplying pairs of sequences as input,
or if the sequence for each datapoint has some associated non-sequence
features, you could concatenate those to the output of the ``FastConv1d``
feature extractor.

For each feature extractor, you can supply additional arguments to control
what kind of features it generates. More details are below.

.. autoclass:: xGPR.FastConv1d
   :special-members: __init__
   :members: predict
