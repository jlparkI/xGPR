Tutorial -- Tuning hyperparameters
======================================

Let's assume we have a Dataset, created as illustrated in the previous
tutorial. We'll now use it to
tune hyperparameters. All of the currently supported kernels in xGPR have
two (Linear, Poly, GraphPoly) or three (FHTConv1d, RBF, GraphConv1d, Matern)
tunable hyperparameters. All kernels share two tunable hyperparameters:
*lambda*, which measures how "noisy" the data is, and *beta* or the 
amplitude, which measures how far the data deviates from the mean. If you
call ``my_model.get_hyperparams()``, these are the first two in the array
that is returned.

It's possible to write your own routine for tuning hyperparameters. For some
pointers on how to do this, see the "Custom hyperparameter tuning" section
under the main page. You should probably first consider the
built-in routines, however, which are easier to use.

There are three major approaches for choosing good hyperparameters that are
built-in to xGPR. Here's a quick look at each, with more details to
follow. There is also an experimental stochastic gradient based technique
(described under the Miscellaneous section on the main page).


.. list-table::
   :header-rows: 1

   * - Approach
     - Description
     - Pros
     - Cons
   * - | "Crude", marginal
       | likelihood
     - | Calculates the marginal likelihood
       | of the data using matrix
       | decomposition.
     - | Fast for small-moderate size datasets when
       | using a small # of random features.
       |
       | Very easy to use.
       |
       | A "quick and dirty" approach.
       |
       | Good way to obtain a starting
       | point.
       |
       | Low risk of overfitting.
     - | Poor scaling with number of RFFs &
       | dataset size.
   * - | "Fine", marginal
       | likelihood
     - | Calculates the marginal likelihood
       | of the data using a
       | scalable approximation.
     - | Good scaling with number of RFFs &
       | dataset size.
       |
       | Low risk of overfitting.
     - | While much more scalable, it is slower
       | than "crude" with a small
       | number of random features.
       |
       | Has more "knobs" that must be set
       | for the approximation to be accurate.
       |
       | Works much better if a good starting
       | point is supplied.
   * - | Validation set
       | performance
     - | Evaluates performance on a
       | supplied validation set.
     - | Scales well to large numbers of random
       | features and large datasets.
     - | Much more prone to overfitting than
       | marginal likelihood
       | methods.
       |
       | While more scalable than "crude",
       | it's fairly slow.
       |
       | Works much better if a good starting
       | point is supplied.

It's worth noting that when tuning using marginal likelihood, we often
don't need that many ``training_rffs`` to achieve good performance.
Refer to the graph in :doc:`Overview</overview>`; notice that
performance on held out data plateaus rapidly as the number of
random features used for tuning increases. Generally, then, it's
not a bad idea to start using a ``crude`` method with 
a small number of ``training_rffs``, then if this is close
to desired performance but not quite there, use a larger number of
``training_rffs`` with approximate marginal likelihood (or validation
set performance). If you are just using a ``crude`` method to
get a "starting point, it is also possible to use it
on a subset of the training data (we'll show you how to do this
shortly).

Let's look at each strategy and how to execute it in more detail. But first...



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
kernels) are pretty generous and seldom need to be moved. Indeed, the lower
bound on ``lambda`` should only be moved with caution, because extremely
small ``lambda`` values may make the model difficult to fit. ``beta``, the
second hyperparameter (shared by all kernels) has more conservative default
bounds, and these may occasionally need to be expanded. The kernel-specific
hyperparameter (for RBF, Matern and convolution kernels) has pretty generous
default bounds but it may rarely be necessary to expand them.

A more common use case is *contracting* the bounds, i.e. setting a smaller search
space so optimization can proceed more efficiently. This is often helpful.
You can after tuning or fitting get the model to generate a bounding box of
a specified width (and to clip that box anywhere it goes past the default
boundaries) by calling:::

  my_new_bounds = my_model.suggest_bounds(box_width = 1, consider_all_shared = False)

The ``crude`` methods are sometimes better at nailing down the best value for the
kernel-specific hyperparameter (lengthscale) than the two shared hyperparameters
*lambda* and *beta*. If ``consider_all_shared`` is True, the box edges for
*lambda* and *beta* will be set to the default bounds for the kernel.


Tuning hyperparameters using marginal likelihood with "crude" methods
----------------------------------------------------------------------

In xGPR, it's possible to tune hyperparameters by maximizing the
marginal likelihood (the probability of the training data). This
strategy has low risk of overfitting, because it automatically
penalizes unnecessary "complexity", so it does not require cross
validations or a validation set.

We can calculate marginal likelihood with the ``crude`` methods,
which use matrix decompositions, or using the iterative ``fine``
methods. The crude methods are often a good starting
point. They are fast if you use a small number of random
features, they can be used on a subset of the training data,
and using our novel strategies, they often converge in
< 20 iterations. Unfortunately, scaling to large numbers of
random features, especially on large datasets (> 1 million datapoints),
with the ``crude`` methods is poor. Thus, it is best to think
of them as "quick-and-dirty" methods that give a good starting
point for further optimization (it often happens however that this
starting point is sufficiently good we do not need to refine it
any further!)

There are currently three ``crude`` methods (plus an experimental
method, addressed under experimental tuning methods on 
the main page of the docs). Our prefered method is this:::

  hparams, niter, best_score, scores = my_model.tune_hyperparams_crude_bayes(my_dataset,
                                     random_seed = 123,
                                     bounds = None, max_bayes_iter = 30,
                                     bayes_tol = 1e-1, n_pts_per_dim = 10,
                                     n_cycles = 3, n_init_pts = 10,
                                     subsample = 1,
                                     eigval_quotient = 1e8,
                                     min_eigval = 1e-5)


This method is a fun twist on Bayesian optimization: we only need *one* pass over
the dataset to acquire *hundreds* of gridpoints along the first two
hyperparameters, so that the Bayesian optimization piece only runs along
the third kernel-specific hyperparameter. This method is a little "finicky",
in that small shifts in things like the optimization boundaries can cause
small shifts in the location of the solution that is ultimately obtained.
This arises from the stochastic nature of the sampling procedure used in
the Bayesian optimization and has negligible impact on performance. If this
bothers you, we suggest using ``crude_grid`` or ``crude_lbfgs`` (both
described below) instead.

Notice that under ``scores``, this function returns a tuple of the kernel-specific
hyperparameter values that were evaluated and the best score associated with each.
Plotting this can sometimes be informative.

If ``bounds`` is ``None``, for this and for all other marginal likelihood
tuning functions, the kernel default hyperparameter boundaries are used.
Otherwise, ``bounds`` should be a numpy array of shape (N, 2) where N is
the number of hyperparameters, the first column is the lower bound, and
the second column is the upper bound. ``bounds`` is always in logspace,
i.e. each entry is the *natural log* of the hyperparameter value. Using
actual values rather than the log of the values is a great way to accidentally
get a *really* strange result. To see how many hyperparameters your kernel
has, use ``my_model.get_hyperparams()`` which will return the log of the current
kernel hyperparameters as a numpy array.

``max_bayes_iter`` controls the maximum number of iterations. Each iteration
involves a single pass over the dataset plus a matrix decomposition. This
matrix decomposition is the main factor limiting the scalability of this
method with increasing numbers of random features; a matrix decomposition
for a 1024 x 1024 matrix is fast, for a 10,000 by 10,000 matrix, not so
much. ``bayes_tol`` is a threshold for convergence, the default is
recommended. ``n_init_pts`` controls how many
points are evaluated before Bayesian optimization starts. The default is
good unless you are using a smaller bounded region, in which case you could
decrease this for greater efficiency.


``n_pts_per_dim`` and ``n_cycles`` controls how many values of the shared
hyperparameters *lambda* and *beta* are considered for each possible kernel-
specific hyperparameter value. Increasing these may lead to a (generally
negligible) boost in performance, but it is almost never necessary -- we
recommend leaving this as default.
recommend leaving this as default. Finally, if ``subsample`` is less than
1 -- if it is 0.1, for example -- this fraction of the training data will
be sampled when tuning hyperparameters. Keep in mind that using more of the
training set will usually improve validation set performance.

``eigval_quotient`` and ``min_eigval`` control how the eigenvalues of the
design matrix are handled and should generally be left as default. Setting
``min_eigval`` to a smaller value (e.g. 1e-6) or ``eigval_quotient`` to
a larger value (e.g. 1e9) can slightly improve performance of this method
but is not usually necessary or recommended.

Another method we have used is this one:::
  
  hparams, niter, best_score, scores = my_model.tune_hyperparams_crude_grid(my_dataset,
                                     random_seed = 123,
                                     bounds = None, n_gridpoints = 30,
                                     n_pts_per_dim = 10, subsample = 1,
                                     eigval_quotient = 1e8, min_eigval = 1e-5)


This method employs the same fun trick as ``crude_bayes``, but rather
than doing Bayesian optimization, it does a gridsearch along the
kernel-specific hyperparameter (if there is one). It's less "finicky" and
than ``crude_bayes``, but usually needs more iterations
to find a good set of hyperparameters. ``n_gridpoints`` determines how
many gridpoints we have along the third, kernel-specific hyperparameter.
This number corresponds to the number of passes over the data, each of
which involves a matrix decomposition as with ``crude_bayes``.
Most of the parameters have the same meaning as for ``crude_bayes`` above.

``n_pts_per_dim`` controls how many values of the shared
hyperparameters *lambda* and *beta* are considered for each possible kernel-
specific hyperparameter value. Increasing these may lead to a (generally
negligible) boost in performance, but it is almost never necessary -- we
recommend leaving this as default.

An alternative to the two strategies above is L-BFGS with multiple restarts.
This is a classic strategy for tuning hyperparameters
of Gaussian processes, although we've found that it often takes 5-10x more
iterations than ``crude_bayes`` or ``crude_grid`` (and thus 5-10x longer).
It is less "finicky" than ``crude_bayes`` but not so foolproof as ``crude_grid``.
As with the other methods, each iteration involves a pass over the
dataset and a matrix decomposition. Here's an example of usage:::

  hparams, niter, best_score = my_model.tune_hyperparams_crude_lbfgs(my_dataset, random_seed = 123,
                                     max_iter = 30, n_restarts = 1,
                                     starting_hyperparams = None,
                                     subsample = 1)

See notes above for most of the parameters.
If ``starting_hyperparams`` is None, a starting point is
selected randomly. As illustrated here, it's possible
to do L-BFGS from a single starting point -- just supply a numpy array
of the starting hyperparams (in logspace, just like bounds) and set n_restarts
to 1. This is sometimes helpful if you know the approximate location
of a good hyperparameter combination. It's *generally* preferable to
set n_restarts to some value greater than 1 so that ``n_restarts``
randomly selected starting locations are used; this gives you a much
better chance of finding the global optimum.

If you've already tuned the hyperparameters using some other method,
``tune_hyperparams_crude_lbfgs`` will use the current hyperparameters as a
starting point, which can occasionally be useful.




Tuning hyperparameters by approximate marginal likelihood
----------------------------------------------------------

The cost of the matrix decompositions used by ``crude`` methods
scale as the cube of the number of random features.
Consequently, these approaches are already slow for 4,000 - 5,000 
random features and are not very useful for any number of random
features much greater than 5,000. It's better to train on GPU
than on CPU, of course, but if you must tune on CPU then this
is even more true, because the matrix multiplications involved
in the ``crude`` procedures will be slow for >> 1024
random features or so.

For a larger number of random features, we can use an iterative
method that approximates the marginal likelihood; these are called
``fine`` tuning methods. The approximation
is actually quite accurate as long as the settings used are appropriate.
A good rule of thumb: Validation set performance after fine-tuning should
almost *never* be worse than validation set performance before fine-tuning,
*unless* we have chosen settings that are causing approximation quality
deterioration, we are not allowing the optimizer sufficient time to
search the space, or the kernel we have selected is completely wrong
for our problem.

Currently there are two supported tuning methods that use this approach,
a Bayesian procedure and a direct (either Powell or Nelder-Mead) procedure.
These approaches are significantly slower for small numbers of
random features and small-
moderate size datasets, but they have better scaling -- they will
scale better as the number of random features and number of datapoints
increases.

These approaches are also less efficient at searching the
whole hyperparameter space. Direct works *much* better
if its starting point is close to the global optimum -- it can get stuck in
local minima. The Bayes routine is good at escaping local minima, but is less
efficient if used to search a large space (e.g. the default search
boundaries). Consequently, it's
often a good idea to get a starting point from a crude procedure using
a small number of random features and / or a subset of the data, then
"fine-tune" using ``fine_direct`` if we are confident our starting point is close
to optimal or ``fine_bayes`` if we are not.

Here are the two currently supported options:::

  hparams, nfev = my_model.tune_hyperparams_fine_direct(my_dataset, bounds = None,
             optim_method = "Powell",
             starting_hyperparams = None, random_seed = 123,
             max_iter = 50, nmll_rank = 1024, nmll_probes = 25,
             nmll_iter = 500, nmll_tol = 1e-6,
             pretransform_dir = None,
             preconditioner_mode = "srht_2")

  hparams, nfev = my_model.tune_hyperparams_fine_bayes(my_dataset, bounds = None,
             random_seed = 123, max_bayes_iter = 30, tol = 1e-1,
             nmll_rank = 1024, nmll_probes = 25, nmll_iter = 500,
             nmll_tol = 1e-6, pretransform_dir = None,
             preconditioner_mode = "srht_2")



As you can see, there are a lot more available "knobs" to turn! This
may look a little intimidating, but the good news is that many of
these can be left at default most of the time (default values are
shown). Notice that for
``fine_direct`` we can supply a starting point (if none is supplied
and the model has already been tuned, it will use the existing
hyperparameters). ``fine_direct`` needs a decent starting point to work
well. ``fine_bayes`` does not start from a single point, so it does not
accept starting hyperparameters. It does however accept bounds,
and it's a really good idea to provide it with some bounds that are
narrower than the default xGPR boundaries, which are too wide of a space
for ``fine_bayes`` to search efficiently. See "A quick note on 
tuning hyperparameter bounds" above to see how to get xGPR to
suggest bounds for you after an initial round of ``crude`` tuning.

``pretransform_dir`` defaults to None. If it's *not* None, it should
be a valid filepath to a location where xGPR can temporarily store
random features when they are generated (xGPR will cleanup when it's
done). On each pass over the dataset, it then loads the saved random
features instead of generating them. This isn't necessarily faster --
it can be slow, because loading large quantities of data from disk
obviously is not fast. This may be faster than generating
random features on the fly *if*:

#. You are using a convolution kernel (especially if the sequences /
   graphs are long or large).
#. You are tuning on CPU (in this case, using a pretransform_dir
   is *highly* recommended.
#. You have an SSD.

Using a pretransform_dir is problematic if the number of random
features you are using * the number of datapoints * 4 (bytes per
float) is on par with your available disk space.

To understand the other options, note that the marginal likelihood
approximation is iterative -- it loops over the dataset repeatedly
to build an approximation, using a preconditioner to reduce the
number of iterations required. The *larger* the preconditioner rank
(nmll_rank), the slower the preconditioner is to construct, but
the more of a speedup it will give to the iterative procedure,
which will then converge faster. A larger nmll_rank also helps ensure
approximation quality, using a rank that is too small can lead to
a drop in approximation quality. We have found ``nmll_rank = 1024``
to be adequate in most cases if using 'srht_2';
``nmll_rank = 2000`` is slower but better for noise-free data (where
the error in model predictions is small), cases where the search
region includes very small values for ``lambda``, and cases where
you are using 'srht' instead of 'srht_2'.

``preconditioner_mode`` can be one of "srht" or "srht_2". "srht"
is faster (requires one pass over the dataset) but lower-quality
(you may need to increase nmll_rank to a larger number to get the
same speedup from the preconditioner).

``nmll_tol`` is the convergence tolerance for the iterative procedure.
Tighter leads to better approximation quality. For noisy data,
1e-6 (the default) is sufficient. For data that is nearly noise-free,
1e-7 is better. ``nmll_iter`` is the maximum number of iterations
allowed (usually this will not be reached in practice, so it's not
a bad idea to set it to a high number like 500).

Finally, ``nmll_probes`` controls the accuracy of log-determinant
approximation. Larger numbers
tend to improve accuracy but with diminishing returns and with
increased cost. 25 is usually fine in our experiments.

``fine_bayes`` is better if the starting
point is far from the optimum, because it's a global procedure. It may
require 30-80 iterations to get a good result. ``fine_direct`` works very
well if the starting point is decent, and may be able to improve
substantially on the starting point in just 25-50 iterations.

Overall, tuning with approximate marginal likelihood is trickier than
either crude or validation set tuning -- there are some knobs we can turn
to affect the quality of the approximation. However, it can improve on the
best performance achieved by crude by enabling you to use a
larger number of random features for hyperparameter tuning, and it is
much less prone to overfitting than validation set tuning.
We have generally found it unnecessary to use *significantly* more
than 10000 random features for tuning -- while you can achieve
better results that way, the gains in performance will be quite small.


Tuning hyperparameters by validation set performance
----------------------------------------------------

We can also tune hyperparameters using performance on a validation
set, or using cross-validation. This is really only recommended if
you have a good validation set to work with; cross-validations are
slow. It's easy to write your own function to do this, or you can
use the ``tuning_toolkit`` in xGPR.

In this approach, the model is fitted repeatedly using different hyperparameter
combinations and the hyperparameters are optimized using Bayesian optimization
or Direct (Powell or Nelder-Mead). Direct needs a good starting point to work
well, Bayesian needs good boundaries to work well. Searching the whole space using
these strategies is not impossible but is expensive.

To execute this approach using either Bayes or Nelder-Mead (choose only
one obviously):::

  from xGPR.tuning_toolkit.direct_fitting_optimizer import Direct_Fitting_Optimizer
  from xGPR.tuning_toolkit.bayesian_fitting_optimizer import Bayesian_Fitting_Optimizer

  hparams = Direct_Fitting_Optimizer(my_model, train_dataset, bounds,
                       optim_method = "Powell",
                       max_feval = 25, validation_dset = None,
                       preset_hparams = None, random_state = 123,
                       score_type = "mae", tol = 1e-3,
                       cg_tol = 1e-6, verbose = True,
                       pretransform_dir = None,
                       mode = "cg")
  
  hparams, cv_hparams, scores = Bayesian_Fitting_Optimizer(my_model, train_dataset, bounds,
                       max_feval = 25, validation_dset = None,
                       preset_hparams = None, random_state = 123,
                       score_type = "mae", tol = 1e-3,
                       cg_tol = 1e-6, verbose = True,
                       pretransform_dir = None,
                       mode = "cg")


Aside from the datasets and model, the options shown are the defaults.
Notice that you have to pass a valid model object you have created,
and that both methods require you to specify a set of boundaries -- this
will be an *N x 2* numpy array where *N* is the number of hyperparameters
(to see what this is for your chosen kernel, run ``my_model.get_hyperparams()``).
To get the model to suggest a good set of boundaries for you, if you've already
done some preliminary tuning with the model, see "A quick note on hyperparameter
tuning bounds" above. Also note that in this case, the model is using
``fitting_rffs`` and not ``training_rffs``, because it is being fitted repeatedly,
so be sure to set ``fitting_rffs`` to what you want it to be.

``score_type`` can be one of "mae" or "mse", indicating to optimize
mean absolute or mean squared error. "max_feval" controls the max
number of function evaluations. ``cg_tol``, ``pretransform_dir`` and
``mode`` control control how fitting is performed on each iteration.
(For more details on these, see the :doc:`Fitting tutorial</fitting_tutorial>`.

In this case we've supplied a validation dataset, which must be an OnlineDataset
or OfflineDataset just like our training set. If we specify ``None`` instead,
the optimizer will use cross-validations, which are more expensive. If
``preset_hparams`` is None, a starting point will be randomly chosen --
this is fine for Bayesian optimization, but will slow things down greatly
for Direct (for Direct, obtaining a good starting point is a good
idea). For Direct, we can specify an ``optim_method`` that is either
"Nelder-Mead" or "Powell". Nelder-Mead tends to be better but slower
(many more iterations to converge).

These approaches are highly scalable (they scale well with increasing
dataset size and/or number of random features) but also slower for small
- moderate size datasets. There is a much higher risk of overfitting than
with marginal likelihood based approaches. The hyperparameters of your
model are automatically set to the best result from the tuning
procedure, so you can fit as soon as the procedure finishes.

Finally, notice that the ``Bayesian_Fitting_Optimizer`` also returns all
of the points it evaluated and the corresponding scores, which can
sometimes be informative.


Other methods
----------------------------------------------------

It is also possible to design your own hyperparameter tuning scheme
or to use an experimental SGD-based method; these are described under
Miscellaneous on the main page.



Next steps
----------------------------------------------------

Once you've tuned hyperparameters, you can retrieve them using:::

  hparams = my_model.get_hyperparams()

It's a good idea to save them somewhere, since you can fit a model
using this set of hyperparameters without having to retune by just
passing it to the fitting functions as we will illustrate.

Once you've tuned hyperparameters, you're ready to fit your model. This
is actually much faster, easier and more straightforward than tuning,
but there are some different knobs we can turn to improve speed
and performance. For more, turn to the next tutorial,
:doc:`Fitting a model in xGPR</fitting_tutorial>`.
