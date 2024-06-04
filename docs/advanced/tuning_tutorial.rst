In-Depth -- Tuning hyperparameters
======================================

General observations on hyperparameter tuning
----------------------------------------------

This In-Depth will focus on a tuning procedure which is
(currently) only available for xGPRegression -- tuning by minimizing
the negative log marginal likelihood (what xGPR calls NMLL). This
quantity represents the probability of the training data averaged
over all possible parameter values, so it automatically penalizes
unnecessary "complexity" (i.e. hyperparameter settings that are
likely to overfit). It correlates very well in our experience
with validation set performance.

Tuning using NMLL is highly advantageous when possible because 1)
no validation set is required and 2) this procedure tends to be
fairly resilient to overfitting, so it enables us to fit kernel
machines to small datasets.


A quick note on hyperparameter tuning bounds
----------------------------------------------

If you don't supply bounds when tuning hyperparameters, xGPR uses default
boundaries for each kernel. You can see what these are after tuning or fitting
a model by calling:::

  default_bounds = my_model.kernel.get_bounds()

this will return the log of the boundaries.

If you find that optimization is suggesting hyperparameter values which are
close to or on the bounds, you might want to set new more generous bounds.
The default bounds for ``lambda`` (the first hyperparamter, shared by all
kernels) are pretty generous and seldom need to be moved. The kernel-specific
hyperparameter (for RBF, Matern and convolution kernels) has pretty generous
default bounds but it may rarely be necessary to expand them.

A more common use case is *contracting* the bounds, i.e. setting a smaller search
space so optimization can proceed more efficiently.


Approximate vs exact NMLL
-----------------------------

All models in xGPR use the random features approximation; thus, all NMLL
calculations are approximate. NMLL calculations in xGPR can however
achieve good scalability for large ``num_rffs`` by introducing an
additional, highly accurate approximation for log determinants. Thus
there are two methods for calculating NMLL in xGPR: ``exact`` and
``approximate``. ``exact`` is much faster if the number of RFFs is
small and (on GPU) is still reasonably fast for even as many as
8,192 RFFs. For larger numbers of RFFs it will however exhibit
cubic scaling in time, quadratic scaling in memory and slow
down dramatically. ``approximate`` is slower for small numbers
of RFFs but has much better scaling.

The quality and speed of the ``approximate`` NMLL calculation
can be controlled by supplying a dictionary, ``manual_settings``,
containing various options when calling ``my_model.approximate_nmll()``
or ``my_model.tune_hyperparams()``. The defaults usually work well,
but you may want if interested to take more control yourself. For
more details, see below.

In addition to calling ``my_model.exact_nmll`` and ``my_model.approximate_nmll``,
you can also use two build-in hyperparameter tuning functions with either
approximate or exact NMLL used for optimization. For details see below.

NMLL tuning methods
----------------------

.. autoclass:: xGPR.xGPRegression
   :members: exact_nmll, approximate_nmll, tune_hyperparams_crude, tune_hyperparams
