Experimental & custom hyperparameter tuning methods
=======================================================

Experimental
----------------------------------------------------

You can also tune hyperparameters using the AMSGrad stochastic gradient
descent method. Note that this is actually a biased estimator of the
full gradient of the marginal likelihood, so it usually achieves a
suboptimal result. That's why this is an experimental method that is
not recommended, so use at your own risk. It is of
course possible to fine-tune the result achieved by sgd using some
other method, and we've achieved good results this way on some problems.
As a compensating advantage, sgd is highly scalable.

Here's an example of how to use it:::
  
  my_model.tune_hyperparams_sgd(my_dataset, random_seed = 123,
                                     n_epochs = 10, minibatch_size = 1000,
                                     lr = 0.02, n_restarts = 5, bounds = None,
                                     start_averaging = 9, nmll_method = "approx",
                                     nmll_rank = 1024, nmll_probes = 25,
                                     nmll_iter = 500, nmll_tol = 1e-6)

Notice that here we additionally specify the initial learning rate as
``lr``, number of epochs and minibatch size, as well as a number of
restarts. All of these
affect performance, and stochastic gradient descent is a lot less
forgiving than ``minimal_bayes`` or L-BFGS (unfortunately). Note that
minibatch sizes > 2000 will be quite slow and are not recommended.

Notice that you can also specify an ``nmll_method`` that is either
``approx`` or ``exact``. At the *end* of each sgd restart, the module
will calculate the marginal likelihood using the method you specify.
``exact`` does not scale well to large #s of
random features -- it is useful up to 5,000 random features or so.
For larger numbers of random features, consider using ``approx``.
To understand the other settings and when to change them, see
the section above on tuning using approximate marginal likelihood.

SGD is mostly useful to find a starting point for another method
(e.g. L-BFGS). Sometimes the result from sgd will be good enough it
does not need any further tuning but this is not often true.


Designing your own
--------------------

You can design your own hyperparameter tuning method and call
the appropriate xGPR methods as well.

If you want to tune using validation set performance, it's easy --
follow the :doc:`fitting tutorial</fitting_tutorial>` to fit the model
once for each set of hyperparameters you specify, then follow
the :doc:`prediction tutorial</prediction_tutorial>` to make predictions
for the validation dataset for each point of interest.

To build a hyperparameter tuning scheme that uses marginal log
likelihood, first, call:::

  bounds = my_model._run_pretuning_prep(dataset, random_seed,
                    bounds, "approximate")

or:::

  bounds = my_model._run_pretuning_prep(dataset, random_seed,
                    bounds, "exact")
  

depending on whether you plan to use exact or approximate marginal
likelihood. ``_run_pretuning_prep`` will set the Dataset to use
the same device that the model is using (datasets otherwise default
to CPU) and create a kernel object. For the ``bounds`` argument, you
can pass ``None`` or an appropriate numpy array if desired. If you
pass ``None``, the call will return the kernel default boundaries;
you can use these or disregard them as you prefer.

Next, call one of the following functions for each set of
hyperparameters you want to evaluate:::

  nmll = my_model.exact_nmll(hyperparams, dataset)
  nmll, gradient = my_model.exact_nmll_grad(hyperparams, dataset,
            subsample = 1)
  nmll = my_model.approximate_nmll(hyperparams, dataset, max_rank = 1024,
                      nsamples = 25, random_seed = 123, niter = 1000,
                      tol = 1e-6, pretransform_dir = None,
                      preconditioner_mode = "srht_2")

See the discussion under the :doc:`tuning tutorial</tuning_tutorial>` under
the approximate marginal likelihood section for more details on the
inputs to ``approximate_nmll``. These control fitting speed and
approximation quality.

All three functions take a numpy array ``hyperparams``, which must be the
same length as the current set of kernel hyperparameters (call
``my_model.get_hyperparams()`` to see these), and a Dataset object.
You can therefore wrap one of these in for example a Scipy optimizer
if desired. All three functions use the current setting for ``training_rffs``.

It is important to note that ``exact_nmll`` calculates the marginal likelihood
using matrix decompositions and hence cubic scaling in the number
of random features. There may also be slight variability in the results
from ``exact_nmll`` when using different linear algebra libraries for
hyperparameter values that result in an ill-conditioned design matrix.
``exact_nmll`` will be very slow for a large number of
``training_rffs``, and is probably not recommended for anything much larger
than 4000 on GPU (on CPU, even less). ``exact_nmll_grad`` calculates the
gradient using matrix decompositions and has the same issue.
``approximate_nmll`` is much more scalable but it does the same amount of
work as fitting the model, so if you call ``approximate_nmll`` 25 times,
expect this to take roughly as long as fitting the model 25x.

Finally, once you are done, you can call:::

  my_model._post_tuning_cleanup(dataset, hyperparams)

This will set the model to use the numpy array that you pass as ``hyperparams``
as its current set of hyperparameters, and now you are ready to fit the model.
