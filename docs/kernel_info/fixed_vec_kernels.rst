Kernels for fixed-vector input
-------------------------------

These kernels handle fixed vector input, similar to a fully-
connected NN. To use one of these, when initializing the
model, set ``kernel_choice = 'kernel name'``, e.g.
``kernel_choice = "RBF"``. See :doc:`Getting started</initialization_tutorial>`
for details.


.. list-table:: Fixed-vector kernels
   :align: center
   :header-rows: 1

   * - Kernel Name
     - Description
     - kernel_specific_params
   * - RBF
     - | Models smooth, infinitely differentiable
       | functions.
     - | "intercept": bool If True,
       | fit a y-intercept.
       | Defaults to True.
   * - Matern
     - | Models "rougher" functions than RBF.
       | nu = 5/2 is twice differentiable,
       | nu=3/2 is once.
     - | "matern_nu":float
       | "intercept": bool If True,
       | fit a y-intercept.
       | Defaults to True.
   * - Linear
     - | Equivalent to Bayesian linear regression.
       | Use "intercept" to indicate if a y-
       | intercept should be fitted.
     - | "intercept":bool
   * - Poly
     - | Approximates polynomial regression.
       | Currently allows degrees 2-4 (higher
       | is likely better handled by an
       | RBF kernel).
     - | "polydegree":int
       | "intercept": bool If True,
       | fit a y-intercept.
       | Defaults to True.
   * - MiniARD
     - | Same as RBF, but rather than having one
       | lengthscale shared between all features,
       | applies different lengthscales to different
       | groups of features.
     - | "split_points":list
       | "intercept": bool If True,
       | fit a y-intercept.
       | Defaults to True.

The ``Linear`` kernel is equivalent to Bayesian linear regression.
If ``intercept`` is False, it will be fitted without a y-intercept
(generally it is preferable to set ``intercept`` to True).

The ``MiniARD`` is an RBF kernel that assigns a different lengthscale
to different kinds of features. You might have data, for example,
where some features are one-hot encoded and others are real. If
so, you could use MiniARD and "learn" a different lengthscale for
each type of feature. For this kernel, supply a list under
``kernel_specific_params`` when creating a model, e.g.:::

  my_model = xGPRegression(training_rffs = 2048, fitting_rffs = 8192,
                        variance_rffs = 512, kernel_choice = "MiniARD",
                        device = "gpu", kernel_specific_params =
                        {"kernel_specific_params":[21,36])

The features in between two split points all share a lengthscale. In this
case, for example, features from 0:21 in the input would share one
lengthscale, features from 21:36 would share another, and features from
36: would share another lengthscale (0 and len(feature_vector) are automatically
added to the beginning and end of split_points). This technique can be
very powerful but also does make tuning more complicated and much slower,
especially if the number of lengthscales is very large, so use judiciously.
The lengthscales learned by ``MiniARD`` during tuning can be used as crude
measures of relative importance (larger = more important group of features).

The chart below illustrates how Matern and RBF fit a simple
"noisy sine wave" dataset. (The arc-cosine kernel and ERF-NN
kernel were implemented originally in xGPR but have been removed
since we have not found many cases where they provide a clear
benefit.)

.. image:: images/toy_data.png
