Experimental & custom hyperparameter tuning methods
=======================================================


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
