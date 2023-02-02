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
     -
   * - Matern
     - | Models "rougher" functions than RBF.
       | nu = 5/2 is twice differentiable,
       | nu=3/2 is once.
     - | "matern_nu":float
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

The ``Linear`` kernel is equivalent to Bayesian linear regression.
If ``intercept`` is False, it will be fitted without a y-intercept
(generally it is preferable to set ``intercept`` to True).

The chart below illustrates how Matern and RBF fit a simple
"noisy sine wave" dataset. (The arc-cosine kernel and ERF-NN
kernel were implemented originally in xGPR but have been removed
since we have not found many cases where they provide a clear
benefit.)

.. image:: images/toy_data.png
