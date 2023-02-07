Tutorial -- Fitting a model
======================================

Once you've tuned hyperparameters (or if you already
in advance have hyperparameters which you know will work
well from previous experiments), you're ready to fit.
Fitting is faster, easier and more foolproof than tuning,
but there's still some knobs we can turn to make it as
fast as possible.

Pretransforming data and when to do it
---------------------------------------

The first option is *pretransforming the data*. This means
generating all of the random features for the training set and
saving them on disk, so they only have to be generated once.
Depending on your hardware and the size of the dataset, this
can be either beneficial or counterproductive.

Although we don't recommend training on CPU,
for CPU, pretransforming data is generally much faster and
is recommended.
For GPU, if the number of features for each input datapoint
is small, loading the data to memory and generating features
on the fly can actually be *faster* than pre-generating the
features and loading them from disk, just because loading them
from disk is slow. The exception is convolution kernels, for
which pretransformation speeds things up even on
GPU, especially if the sequences are long / graphs are large,
in which case pretransform is highly recommended.
Of course, if your hard drive happens to be an SSD, this
can make the pretransformation approach much faster.

There is one other consideration, which is disk space. If you're
using ``fitting_rffs = 32768``, for example, this means that we
generate 32,768 random features for each training set datapoint.
If we pretransform, we're saving those 32,768 features somewhere
on disk. For 1 million datapoints, with 32,768 random features
each, we're looking at 131 GB of disk space already, which may be
ok depending on hardware. If you have 10 million datapoints,
however, 1.3 TB might be more problematic! In this case, you're
better off generating the features as needed for each minibatch
during fitting, as opposed to using up an absurd amount of disk
space to store pregenerated features.

If given all of the above you decide it makes sense to pretransform,
you do it like this:::

  pretransformed_dataset = my_model.pretransform_data(training_dataset,
                  pretransform_dir = "~/temp_dir", random_seed = 123,
                  preset_hyperparams = None)

The pretransformed data will be saved as numpy arrays under whatever
directory you supply as ``pretransform_dir``. Note that you can use
``None`` for hyperparams if the model was just tuned. Otherwise,
if you have a specific set of hyperparameters you'd like to use, you
*must* supply them to ``pretransform_data``, because
``pretransform_data`` will use those hyperparameters when generating
the random features. If not ``None``, the hyperparams should be
a numpy array of shape (N) for N kernel hyperparameters. To see how
many hyperparameters your kernel has, use ``my_model.get_hyperparams()``.
Note that you must supply the *natural log* of the hyperparameters,
not the actual values -- supplying the actual values instead of the
natural log can lead to fairly strange results. (All xGP_Regression
functions you can call to retrieve hyperparameters return the log
of the hyperparameters unless you specify otherwise.)

Once you have your pretransformed_dataset, you can use it in all
subsequent fitting steps in place of your training set.
Once you're done with your pretransformed dataset at the end of fitting,
you can clean up the files associated with it as follows:::

  pretransformed_dataset.delete_dataset_files()


Preconditioning
----------------

Preconditioning is the heart and core of how model fitting is done
in xGPR. It's possible to fit without preconditioning, but it's
generally going to be much slower. Building a preconditioner
requires either one or two passes over the dataset.

To build a preconditioner, run:::

  preconditioner, ratio = my_model.build_preconditioner(my_dataset,
                     max_rank = 512, random_state = 123,
                     method = "srht_2", preset_hyperparams = None)

Once again, if you have hyperparameters you'd like to use for fitting
that do not result from tuning ``my_model`` on this dataset, you
should supply them as ``preset_hyperparams``.

``method = 'srht'`` uses a fast Hadamard transform based construction
with a single pass over the dataset. ``method = 'srht_2'``, by contrast,
runs two passes over the dataset and does involve some matrix
multiplications, so it is more expensive, but the resulting preconditioner
is usually better. A preconditioner built with 'srht_2' will usually reduce
the number of iterations required to fit by about 20-25% compared with one
built with 'srht'. In general, then, prefer 'srht_2'. If you are training
on CPU, 'srht' may be preferable simply because of its greatly reduced
cost however.

The ``ratio`` is a pretty good predictor of how many iterations it will
take xGPR to fit. See the following chart, where ``ratio`` is referred
to as beta / lambda^2 :

.. image:: images/ratio_vs_iter.png
   :width: 600
   :alt: Ratio vs iterations to convergence

Smaller ratios in general mean fewer iterations are required to fit. Each iteration
is a single pass over the dataset (similar to an epoch when training a deep learning
model but generally much faster). The ratio in turn is a dataset-dependent
function of ``max_rank``. Notice that the relationship between ratio and
number of iterations is different for ``srht`` and ``srht_2``; ``srht_2`` is
better but as noted requires two passes over the data to build.

If based on the ratio we obtain, we think that fitting
will take more iterations than we would like, we can reconstruct the preconditioner
using a larger ``max_rank`` and get a better ratio (doubling ``max_rank`` is
usually a good step to take). Bear in mind however that large values of
``max_rank`` are also expensive, because preconditioner construction requires
a matrix decomposition, which is cheap for say 512 x 512 or 1024 x 1024, but
not so much for very large ``max_rank``. Therefore the goal is to push
``max_rank`` high enough to get an acceptable likely number of iterations,
but not so high that it is unnecessarily expensive to construct. In our
experiments, we have rarely needed to take ``max_rank`` higher than 2000,
but it is of course possible to do so if necessary.

Note that the chart above was constructed using fits with ``tol=1e-6``. The number
of iterations will be greater if you select a tighter ``tol`` and less with a
looser ``tol``. Also note that for very small values of the ``lambda`` hyperparameter
(shared by all kernels, and the first hyperparameter that is returned
from ``model.get_hyperparams()``) is very small, the ``ratio`` may 
dramatically overestimate how many iterations are required to fit. See the 
molecule example on the main page for a case where the ratio is > 1000 but the model
fits in << 100 iterations! Consequently, the graph above should
only be used as a rough guide. When in doubt, a quick experiment on a subset of the
data may prove helpful in deciding what ``ratio`` is "good enough" before 
building a preconditioner on the full dataset and fitting.

Fitting
--------

Our favorite method for fitting in xGPR is preconditioned conjugate
gradients, which is as follows:::
  
  my_model.fit(training_dataset, preconditioner = my_new_preconditioner,
                tol = 1e-6, max_iter = 500,
                random_seed = 123, run_diagnostics = False,
                mode = "cg", preset_hyperparams = None)

Once again, if you have hyperparameters you'd like to use for fitting
that do not result from tuning ``my_model`` on this dataset, you
should supply them as ``preset_hyperparams``, for both this and all
other fitting modes.

You can run without a preconditioner by setting ``preconditioner = None``,
but for CG at least, this is not recommended.
                
If run_diagnostics is True, fitting will return the number of iterations,
a list of the remaining error on each iteration and other occasionally
useful information. ``max_iter`` places a ceiling on the max allowed
number of iterations. Finally, ``tol`` is the threshold at which
the fit has converged; the smaller the threshold, the tighter the fit.
Using a smaller value for tolerance always improves model performance
but with sharply diminishing returns, and it increases the number of
iterations required to fit. (The chart for ratio vs iterations above
was generated using ``tol = 1e-6``; ``tol = 1e-7`` will require more
iterations, ``tol = 1e-5`` fewer).

We recommend ``tol = 1e-6`` as
a default if using 32,768 random features or less.
For noisy data, if you're in a hurry, ``tol = 1e-5``
often gives almost equivalent results.
For relatively noise-free data, where the model is already highly
accurate and we would like it to be even more so, ``tol = 1e-7``
is recommended. ``tol = 1e-8`` is usually expensive overkill,
unless the data is nearly noise-free and you are using a large number
of random features, in which case it may be worthwhile to make ``tol``
smaller. When in doubt, a quick experiment on a subset of the training
data may often prove helpful.

If ``fitting_rffs`` is small (e.g. 2048), you can fit using a single
pass over the data, no preconditioner required! by using ``mode = exact``,
for example:::

  my_model.fit(training_dataset, random_seed = 123, mode = "exact")

this will use a Cholesky decomposition to fit.

Another preconditioner-free approach is L-BFGS. This one generally requires
a much tighter tolerance than CG (e.g. ``tol = 1e-10``) 
to get the same result, and may
require a large number of iterations, so it's not recommended for
anything except small datasets, where iterations are relatively cheap.
With that said, it works quite well for small datasets and
neither requires nor uses a preconditioner, so on a small dataset it
may be a good default. To fit this way, use:::

  my_model.fit(training_dataset, random_seed = 123, mode = "lbfgs")

We also include a couple of stochastic gradient descent methods.
One of these, ``mode = "amsgrad``, is fairly limited; it does not
require a preconditioner (and will ignore it if you supply one),
but it is seldom able to achieve much better than ``tol = 1e-3``,
which may be adequate for very noisy data, but is still definitely
not optimal.

More useful is ``mode = sgd``, which uses stochastic
variance reduced gradient descent (SVRG). Without preconditioning,
this method too is very bad, so we recommend
always using a preconditioner for SVRG. To fit using SVRG, use:::

  my_model.fit(training_dataset, random_seed = 123, mode = "sgd",
                preconditioner = preconditioner, manual_lr = None,
                tol = 1e-6, max_iter = 500)

Note that ``max_iter`` in this case is the maximum number of epochs;
all other arguments have the same meaning as for CG. If ``manual_lr``
is None, SVRG will try to automatically select a good learning rate;
if a float is supplied, it will use that as the starting learning
rate instead. THe automatic learning rate selection is fairly limited
and you *may* be able to do better yourself using careful learning
rate tuning (e.g. fitting with ``max_iter = 2`` using different learning
rate settings), although this can be a lot of work.

In general, we've found that SVRG with preconditioning can like CG
achieve very tight tolerances if needed, but CG in all of our
experiments was much more efficient, requiring many fewer iterations
and of course with no learning rate tuning required. We therefore
prefer CG.

Finally, if you're fitting a model as part of some hyperparameter
tuning scheme, you can supply the argument ``suppress_var = True``
to avoid calculating variance (since you won't need it). This
saves a single additional iteration over the dataset.

To see how to make predictions and cluster or visualize data
with a fitted model, continue to :doc:`Making predictions</prediction_tutorial>`.
