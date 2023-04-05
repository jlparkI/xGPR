Getting started
======================================

Building a dataset
---------------------

To *make predictions with a trained model*, you can feed xGPR a
numpy array (either 2d for fixed-vector kernels or 3d for sequence and
graph kernels). To *train the model and/or perform
convolution-based feature extraction*, by contrast, you have to supply your
data in the form of a Dataset object, which is
analogous to the DataLoader in PyTorch.

Currently you can build a Dataset using either a numpy X array and a y-array
in memory *OR* using a list of x-files and a corresponding list of y-files,
both of which must be numpy arrays saved on disk as .npy files. In the latter
case, xGPR will only load one x-y pair from the list at a time, making it easy
to work with datasets too large to fit in memory. The order in which datapoints
are provided is not important, *unless* you plan to use stochastic gradient
descent for hyperparameter tuning or fitting, in which case datapoints should
be "shuffled".

One important note about Datasets in xGPR -- xGPR by default
finds the mean and standard deviation of training y-values while
constructing a Dataset and rescales
y-values based on this while it's training. Predictions are rescaled
using the training-y mean and standard deviation so that they are on
the original, expected scale, so these manipulations are largely
invisible to the user. It's still good to be aware of this (so you
don't feel the need to rescale the y-data yourself). The input x-values,
on the other hand, are *not* rescaled -- if it's important to
do this for your type of data, you'll need to do that part yourself.

Let's say you have a numpy array X of features and a corresponding 1d numpy
array Y of regression labels. This could be either a 2d N x M array
with N datapoints and M features or a 3d array with N datapoints, M timepoints or
tokens and D features per token / timepoint.
To build a Dataset, do the following:::

  from xGPR.data_handling.dataset_builder import build_online_dataset
  
  my_dataset = build_online_dataset(X, Y, chunk_size = 2000)



Alternatively, let's say you have 100 numpy files (.npy) on disk, each of which
is an N x M array of features, and 100 corresponding numpy files that are 1d
arrays containing the regression labels. The filepaths to these files (can be
relative or absolute) are stored in two lists called xfiles and yfiles. In this
case, to build a dataset, do the following:::
  
  from xGPR.data_handling.dataset_builder import build_offline_fixed_vector_dataset
  
  my_dataset = build_offline_fixed_vector_dataset(xfiles, yfiles,
                  chunk_size = 2000, skip_safety_checks = False)


The safety checks are a little slow -- they consist in loading
each xfile and yfile, checking to make sure it doesn't contain ``np.inf``,
``np.nan`` or any very large numbers, and that each
xfile has the same number of datapoints as the corresponding yfile.
If you're *confident* your data is clean, you can make dataset construction
faster by setting ``skip_safety_checks`` to True. (Note, however, that if some
of your arrays *do* have problems and you skipped safety checks, you run
the risk of some really interesting errors during training.)

Finally, let's say you have the same lists of numpy files, but the x-arrays are each
3d -- they are sequences, graphs or multivariate time series. In this case, do
the following:::

  from xGPR.data_handling.dataset_builder import build_offline_sequence_dataset
  
  my_dataset = build_offline_sequence_dataset(xfiles, yfiles, chunk_size = 2000,
                  skip_safety_checks = False)


The Dataset objects returned by any of these three `dataset_builder` functions
can be used interchangeably -- they only differ in their construction.

Note the `chunk_size` parameter, which we've here set to the default of 2000.
`chunk_size` controls how many datapoints xGPR will work with at a given time.
If your data is a list of numpy arrays on disk, `dataset_builder` will check
to make sure that each file contains *at most* `chunk_size` datapoints, so when
saving your numpy arrays to disk you should make sure that each contains at
most the same number of datapoints you plan to use for chunk_size. If it's
data in memory, the Dataset will only provide the data back to xGPR in chunks
of this size. `chunk_size` helps control memory consumption -- doubling it
may make xGPR slightly faster but will close to double the memory consumption,
while halving it will do the reverse.

Now it's time to set up a model.


Setting up a model for fixed-length vector data
-------------------------------------------------

Start by creating a model:::

  from xGPR.xGP_Regression import xGPRegression
  my_model = xGPRegression(training_rffs = 2048, fitting_rffs = 8192,
                        variance_rffs = 512, kernel_choice = "RBF",
                        device = "gpu", kernel_specific_params =
                        {"matern_nu":5/2, "conv_width":9, "polydegree":2},
                        verbose = True, num_threads = 2)


``kernel_choice`` is the kernel. For a list of options, and for help choosing, see
the Available Kernels section on the main page.

``num_threads`` controls how many threads are used for random feature generation
when running on CPU (if running on GPU, it is ignored). Setting this to a larger
number than what can execute in parallel on your CPU will actually slow things down,
so if in doubt, use the default.

``kernel_specific_params`` is a dictionary of parameters specific to certain
kernels, e.g. Matern and 1d convolution. You only need to supply this
if you're working with one of those kernels (and if you're not happy with
the default values). ``device`` determines whether xGPR will try to fit on
GPU or CPU; you can change this at any time using ``my_model.device = "cpu"```
(or "gpu", as appropriate). The remaining options determine the accuracy of the
approximation and its computational cost.

Let's digress for a minute and discuss how these options affect results.

Choosing the number of RFFs; a brief digression
-----------------------------------------------

xGPR is an approximation to an exact Gaussian process. The larger the number of
RFFs, the better the approximation, and in fact, the error of the approximation
decreases rapidly with the number of RFFs. This means that in general,
we don't need a *huge* number of RFFs to get a decent result. It also means
that increasing the number of RFFs will nearly always improve performance
but with diminishing returns -- going from 4096 to 8192 will yield a bigger boost
than going from 8192 to 16384 and so on. If we get a decent
result and we want it to be a little better, we can double the number of RFFs
we're using. Using more RFFs means xGPR will take longer to train however. To
see how RFFs affect training time and accuracy, see :doc:`Overview</overview>`.

``training_rffs``, ``fitting_rffs`` and ``variance_rffs`` control how accurately
xGPR approximates an exact GP during hyperparameter tuning, model fitting, and
when calculating uncertainty on predictions, respectively. See
:doc:`Overview</overview>` for details.


Setting up a model for convolution
---------------------------------------------

There are currently two ways to do convolution on multivariate sequence
(multivariate time series, sequences) and graphs. The first is to use
a dedicated convolution kernel, (e.g. ``FHTConv1d`` for sequences
or ``GraphRBF`` for graphs), e.g.:::

  from xGPR.xGP_Regression import xGPRegression
  my_model = xGPRegression(training_rffs = 2048, fitting_rffs = 8192,
                        variance_rffs = 512, kernel_choice = "FHTConv1d",
                        device = "gpu", kernel_specific_params =
                        {"conv_width":9}, verbose = True)

Everything else remains unchanged, you just need to ensure the dataset
you supply contains 3d arrays (otherwise a ValueError is raised). For
details on available convolution kernels, see the
Available Kernels section on the main page.

Another option specific to certain kernels is to use a static layer,
then feed the output of this static layer into an ``RBF`` kernel.
To do this, you'll 
need to create a "static layer" object and run your training
dataset through it. This static layer object will now become
part of your pipeline, and when making predictions you'll need to run any
array for which you want predictions through the "static layer"
as well.

Here's how to build and use a static layer on an existing
dataset and for making predictions. In this example, we've
already created a Dataset called ``my_dataset`` that we want
to use for training, and our test data is an array called
``my_xtest``. We illustrate using ``FastConv1d``, currently
the only `static_layer` available in xGPR, which is a type
of kernel for sequences and time series that is completely
different from the ``FHTConv1d`` kernel. It essentially mimics
a three-layer 1d convolutional neural network:::

  from xGPR.static_layers.fast_conv import FastConv1d

  conv_s_layer = FastConv1d(seq_width = 20,
                               device = "gpu", conv_width = [9],
                               num_features = 2048,
                               random_seed = 123)

  #The next line creates my_conv_dataset which we can use for training. 
  my_conv_dataset = conv_s_layer.conv1d_pretrain_feat_extract(my_sequence_dataset,
                                     "~/my_temp_dir")
  #The next line converts a single numpy array of input data into a numpy
  #array we can feed into a trained xGPR model to make a prediction.
  my_new_test_array_conv = conv_s_layer.conv1d_x_feat_extract(my_sequence_xtest,
                                    chunk_size = 2000)

Notice that for training data -- a Dataset we're going to use for training -- we need to supply
a directory where FeatureExtractor can save the results. The returned Dataset
(e.g., ``my_conv_dataset``) can be used for training an xGPR model with
an ``RBF`` kernel. For any arrays where you want to make predictions,
you do not need to supply a directory -- the feature extraction is
performed in memory.

For more on how to choose a kernel or a static layer etc,
see the Available Kernels section on the main page.


Once you've set up a training dataset and a model, you're ready to tune
the kernel hyperparameters. It's possible to write your own hyperparameter
tuning routine, and we'll illustrate how you can do this as well, but xGPR has a
number of built-in approaches that we recommend, and we'll focus on these.
To explore, continue to :doc:`Tuning hyperparameters in xGPR</tuning_tutorial>`.
