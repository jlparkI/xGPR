xGPR quickstart
===============================================

Build your training set
-------------------------

Start by building a Dataset, which is similar to a DataLoader in PyTorch. If
your data is organized in one of a couple fairly common ways, you can
use a built-in xGPR function to build this dataset. If your data is in
some other form(e.g. a fasta file, an SQLite db or an HDF5 file) and you
don't want to make a copy of it, you can instead
subclass xGPR's ``DatasetBaseclass`` and build a
custom ``Dataset`` object. We'll look at the more common
situations first:::

  from xGPR import build_regression_dataset

  reg_train_set = build_regression_dataset(x, y, sequence_lengths = None,
                        chunk_size = 2000)

``x`` and ``y`` can be *either* numpy arrays OR a list of filepaths
to ``.npy`` files saved on disk. If the latter, no single one of the
files can contain more datapoints than the ``chunk_size`` setting.
``chunk_size`` controls how many datapoints xGPR processes at a time
while training. Unlike deep learning, this doesn't affect the fitting
trajectory in any way; it just affects memory consumption. If your
datapoints are really large, you can use a smaller chunk size to
reduce memory consumption.

The ``x`` array(s) can be either 2d for tabular data or 3d for sequences
and graphs. If they are 2d, they should have shape (N, M) for N datapoints
and M features. If they are 3d, they should have shape (N, D, M) for N
datapoints, D sequence elements / timepoints / nodes and M features.
If they are arrays saved on disk, you can save the data as float32 or
even uint8, and xGPR will convert it to float32 when loading (this can
often save considerable disk space and also make model fitting faster).

If your data is 3d (i.e. sequences or graphs), you have to also supply
``sequence_lengths``. If ``x`` is an array with ``shape[0]=N``,
``sequence_lengths`` should also be an array with shape ``(N,)`` that
indicates the length of each sequence (or number of nodes in each graph)
excluding zero-padding (unless you want to include zero-padding in the 
kernel calculation for whatever reason, in which case, just set all
sequence_lengths to be ``x.shape[1]``). xGPR will then "mask" the zero-
padding when doing the kernel calculations.

If your data is 3d but ``x`` is a list of ``.npy`` files on disk,
``sequence_lengths`` should be a list of ``.npy`` files on disk
of the same length. For each ``x`` file with shape ``(N,D,M)``,
the corresponding sequence_length file should be an npy array
of shape ``(N,)`` indicating the length of each corresponding
sequence in that ``x`` file again (typically) excluding zero-
padding.

If using 3d data, you'll also have to pass the sequence lengths of your
test datapoints to ``model.predict`` when making predictions -- see below
for examples.

See the Examples section for some illustrations of working with sequence
and graph data. For fixed-vector 2d data, of course, sequence lengths are not required.

When you create the dataset, xGPR will do some checks to make sure that
what you fed it makes sense. If the dataset is very large, these may take a
second.

Finally, let's say your data is a fasta file, a csv file, an HDF5 file or some
other format. You can create your own Dataset that loads the data in minibatches
during training and does any preprocessing you want to do on each minibatch.
To see how to do this, check out :doc:`notebooks/custom_dataset_example`.

Fit your model and make predictions
-------------------------------------

Let's create two models using RBF kernels and fit them:::

  from xGPR import xGPRegression

  #Notice that we set the device to be "cuda". If there are multiple
  #cuda devices, xGPR will use whichever one is currently active. You can control
  #this by setting the environment variable CUDA_VISIBLE_DEVICES,
  #e.g. "export CUDA_VISIBLE_DEVICES". Also note that for regression,
  #we specify a quantity, "variance_rffs", which must be < num_rffs.
  #This controls the accuracy of the variance approximation when
  #calculating uncertainty on predictions. We've found 512 - 1024 is
  #usually fine.

  reg_model = xGPRegression(num_rffs = 2048,
                        variance_rffs = 512, 
                        random_seed = 123.
                        kernel_choice = "RBF",
                        device = "cuda", kernel_settings = {})

  #Nearly all xGPR kernels have either one or two hyperparameters. We'll
  #talk more about how to optimize hyperparameters shortly.
  my_hyperparams = np.array([-3.1, -1.25])
  reg_model.set_hyperparams(my_hyperparams, reg_train_set)

  #We can change the number of RFFs at any time up until we fit the model.
  reg_model.num_rffs = 2000



  #We can fit using "exact" mode or "cg" mode. "cg" mode is much faster
  #if num_rffs is very large. "exact" is much faster if "num_rffs" is small,
  #e.g. < 3000. We'll use both here just for illustrative purposes.
  #tol controls how tight the fit is; 1e-6 (the default) is usually fine;
  #1e-7 (tighter fit) will improve performance slightly, especially if
  #data is close to noise-free, but is not usually necessary; 1e-8 is
  #usually overkill.

  reg_model.fit(reg_train_set, mode="cg", tol=1e-6)

  #To make predictions, just feed in a numpy array.
  #If we want to switch over to CPU for inference now that the model
  #is fitted, we can do that too, e.g.:

  model.device = "cpu"

  #For regression, predictions are just
  #a numpy array of predicted y-values. chunk_size controls
  #how many datapoints xGPR processes at a time in the input array; larger
  #slightly increases speed but increases memory usage. If you're worried
  #about memory usage, set a small chunk_size, otherwise default of 2000 is fine.

  reg_preds = reg_model.predict(xtest_reg, sequence_lengths = None, chunk_size = 2000)

  #For regression, we can also get the variance on the predictions, which
  #is useful as a measure of uncertainty.

  reg_preds, reg_var = reg_model.predict(xtest_reg, sequence_lengths = None, get_var = True)


And that's basically it!

We used "RBF" kernels here, but there are plenty of other options; see the Kernels
section on the main page. Some of them have options that you can supply under the
``kernel_settings`` dict (e.g. the convolution width if using a sequence kernel).

Notice also that we had to specify ``num_rffs`` when setting up the model (but can
change it subsequently as well, at least right up until we fit). ``num_rffs`` controls
how accurately the kernel is approximated. The error in the kernel approximation falls
off exponentially with larger ``num_rffs`` values, so increasing ``num_rffs`` generally
makes the model more accurate, but with diminishing returns. It also increases
computational expense (fitting using ``num_rffs=1024`` will be much faster than fitting
with ``num_rffs=32,768``).

Finally, notice that when calling ``model.predict`` just as when building a dataset,
``sequence_lengths`` is None if you're using a fixed-length kernel; if you're inputting
a 3d array and using a convolution kernel, you have to supply a numpy array of sequence
lengths so that xGPR can mask zero-padding (if you're using zero-padding).

There's one big missing piece we haven't discussed so far of course, which is...


How to find good hyperparameter values?
----------------------------------------

Most kernels in xGPR have either two hyperparameters ("lambda", "sigma") or one ("lambda").
(There's an exception to this, the ``MiniARD`` kernel, which is a fixed-length kernel
that assigns a different "importance" or lengthscale to different groups of features. 
We'll save that one for an advanced tutorial.) The Lambda hyperparameter is like the ridge
penalty in ridge regression: it provides regularization and is roughly related to how "noisy"
the data is expected to be. Larger (more positive) values = stronger regularization.
xGPR squares the Lambda hyperparameter when fitting.

The "sigma" hyperparameter, for kernels that have it, is an inverse lengthscale that (to oversimplify
a little) determines how close datapoints must be in order to be considered similar. Smaller (more
negative values) cause points that are farther apart to be considered "similar".

xGPR *always* uses *the natural log of the hyperparameters* as input, and internally converts those
to actual values. So if you have:::

  my_model.set_hyperparams(np.array([-1., 0.]), my_train_dataset)

the Lambda value that xGPR will use when fitting is ``(1 / e)^2``, and the sigma value will be ``1``.
(This may seem strange -- it's really just for internal convenience). For numerical stability reasons,
we don't recommend setting Lambda to a value much lower than ``-6.907`` or so
(``(e^-6.907)^2`` is about 1e-6). So for Lambda, it usually makes sense to search across the
range from -6.907 or so to 3 or so. For sigma, the optimal value is usually somewhere in the -7 to 2
range (depending on dataset and kernel).

One simple way to find good hyperparameter values is to fit the model using different 
hyperparameter settings and look at performance on a validation set. So in this scheme, for each set of
hyperparameters you're considering, you would:::

   def my_hparam_evalation_function(my_new_hyperparams, my_validation_set_array,
                    my_validation_set_sequence_lengths = None):
       my_model.set_hyperparams(my_new_hyperparams, my_train_dataset)
       my_model.fit(my_train_dataset, mode="cg")
       preds = my_model.predict(my_validation_set_array, my_validation_set_sequence_lengths)
       ##Add some score evaluation, R^2, MAE, accuracy, etc. here...
       return score

where ``my_new_hyperparams`` is a numpy array. You can easily plug this into Optuna or
some other hyperparameter tuning package, do Bayesian optimization or grid search or
any other procedure you like.

You can tune hyperparameters this way, but for regression, there's
a much nicer way to evaluate hyperparameters, which uses negative log marginal likelihood
(what xGPR calls NMLL). In Bayesian inference, the marginal likelihood is the probability
of the training data averaged over *all possible parameter values*.  A lower NMLL means
a better model (and a higher NMLL means a worse model). The NMLL on the training set in
general correlates *very* well with performance on held-out data. So, for regression we
don't really even need a validation set to tune hyperparameters; we can just calculate the
NMLL for different hyperparameter settings and see which one gives us the best result.

Here's an example:::

   def my_regression_hparam_evalation_function(my_new_hyperparams):
       #If num_rffs is small, use this function
       nmll = my_model.exact_nmll(my_new_hyperparams, my_training_dataset)
       #If num_rffs is large, use this function
       nmll = my_model.approximate_nmll(my_new_hyperparams, my_training_dataset)
       return nmll

Now, we just minimize the value returned by this function -- again, we can use Optuna,
grid search, Bayesian optimization, what have you.

Notice one thing in the function above. ``exact_nmll`` is much faster if the
number of RFFs is small. On GPU, it can be reasonably fast up to about 8,000 RFFs or so.
It has cubic scaling, however, so for large numbers of RFFs it can get very
slow very quickly. ``approximate_nmll`` has much better scaling and so is your
friend if you want to tune using a large ``num_rffs``. It does involve an additional
approximation (above and beyond the random feature approximation used throughout xGPR).
This additional approximation is very good in general but its quality and speed can
be fine-tuned if desired by using some additional knobs; see the Advanced Tutorials for
more.

Finally, for regression, xGPR offers two build-in functions that can do hyperparameter
tuning for you by minimizing the NMLL. These are::

  my_model.tune_hyperparams_crude(my_training_dataset, bounds = None, max_bayes_iter = 30,
                                         subsample = 1)
  my_model.tune_hyperparams(my_training_dataset, bounds = None, max_iter = 50, tuning_method = "Powell",
                            starting_hyperparams = None, n_restarts = 1,
                            nmll_method = "exact")

  #Get the final hyperparameters optimized by tuning as a numpy array
  my_final_hyperparams = my_model.get_hyperparams()


(There are some other knobs we can turn on ``tune_hyperparams``; see Advanced Tutorials for more.)

The first function, ``tune_hyperparams_crude``, is a remarkably efficient way to rapidly
search the whole hyperparameter space for 1 and 2 hyperparameter kernels. It lets you
specify a "subsample" argument; if this is less than 1(e.g. 0.5), it will use the specified fraction
of the training data when tuning. Both functions let you specify search boundaries or just pass
None (the default) for ``bounds``; if None, xGPR uses some default search boundaries.

``tune_hyperparams_crude`` uses an SVD, which means it doesn't scale well
-- it can get pretty slow for  ``num_rffs = 3,000`` or above. Fortunately, we've generally
found that the hyperparameters which give good NMLL with a small number of RFFs
(a sketchy kernel approximation) are usually not too terribly far away from those which give
good NMLL with a larger number of RFFs (a better kernel approximation).
(This is a rule of thumb, and like all rules of thumb should be used with caution.)
So, one way to use these two functions together is to use ``tune_hyperparams_crude`` for a
fast initial search, then (if desired) further fine-tune the hyperparameters using
``tune_hyperparams``. For example:::

  my_model.device = "cuda"
  my_model.num_rffs = 1024
  my_model.tune_hyperparams_crude(my_train_dataset)
  rough_hparams = my_model.get_hyperparams()

  #Use rough_hparams as a starting point for fine-tuning.
  #We could also set a bounding box around rough_hparams,
  #pass that as bounds, set n_restarts to say 3 and
  #thoroughly explore the space around rough_hparams. Or
  #even just do a gridsearch across the space around rough
  #hparams. See the examples section for some illustrations.
  my_model.num_rffs = 4096
  my_model.tune_hyperparams(my_train_dataset, max_iter = 50,
                        tuning_method = "L-BFGS-B",
                        starting_hyperparams = rough_hparams,
                        n_restarts = 1,
                        nmll_method = "exact")

``tune_hyperparams`` can use one of three different algorithms or ``tuning_method``:
``Powell``, ``L-BFGS-B`` and ``Nelder-Mead``. ``L-BFGS-B`` uses the fewest iterations,
but has to calculate the gradient on each, so it's slow if ``num_rffs`` is large.
If ``num_rffs`` is large, instead, consider ``Powell`` and ``Nelder-Mead``. ``Nelder-Mead``
is usually better than ``Powell`` at finding the absolute best possible value, but
it can take a *lot* of iterations to converge, so it's only good if you're not in a
hurry. We generally prefer Powell to Nelder-Mead.

Remember that when calculating NMLL, we could use ``exact_nmll`` or
``approximate_nmll``. The function ``tune_hyperparams`` offers you the same choice:
you can set ``nmll_method`` to either ``nmll_method=exact`` or ``nmll_method=approximate``,
and the considerations are the same. Again, ``exact`` is faster if ``num_rffs`` is small,
while ``approximate_nmll`` has better scaling.

Finally, one important thing to keep in mind. Most of these methods run at reasonable
speed on GPU. On CPU, however, tuning with a large ``num_rffs`` can be a slow slow slog.
Setting the ``num_threads`` parameter on your model can help a little, e.g.:::

  my_model.num_threads = 4

``num_threads`` is ignored if you're fitting on GPU. But that can only help so much. We strongly
recommend doing hyperparameter tuning and fitting on GPU whenever possible. Making predictions,
by contrast, is reasonably fast on CPU. So fitting on GPU and
doing inference on CPU is a perfectly viable way to go if desired.

That's really all you absolutely need to know! For lots of useful TMI, see Advanced Tutorials.
