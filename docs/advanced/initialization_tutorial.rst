Datasets & class constructors: In-depth
========================================

Dataset construction: details
---------------------------------

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
be "shuffled". Likewise, the size of the .npy files (i.e. number of datapoints)
is not important, so long as they are all less than some maximum size.

When loading data, xGPR converts it to float32. You can therefore save it on
disk as float32, float64, or even uint8 or any other convenient format. Saving
it as float32 or (if applicable) uint8 can save considerable memory and make
model fitting faster (since there is much less to load on each pass over the
dataset).

For regression datasets, use:

.. autofunction:: xGPR.build_regression_dataset

And for classification datasets, use:

.. autofunction:: xGPR.build_classification_dataset

Now for some more details on model construction.


Setting up regression and classification models
-------------------------------------------------

.. autoclass:: xGPR.xGPRegression
   :special-members: __init__


.. autoclass:: xGPR.xGPDiscriminant
   :special-members: __init__


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
we're using. Using more RFFs means xGPR will take longer to train however.

``num_rffs`` and ``variance_rffs`` control how accurately
xGPR approximates an exact GP when making predictions and when estimating
uncertainty.

Setting hyperparameters
------------------------

At any time after creating the model we can set its hyperparameters using
``model.set_hyperparams()``. Note however that when we do this we need to
tell the model a few things about what the training data will look like.
The easiest way to do this is to pass it the training Dataset object.

.. autoclass:: xGPR.xGPRegression
   :special-members: set_hyperparams


Setting up a model for convolution
---------------------------------------------

There are currently two ways to do convolution on multivariate sequence
(multivariate time series, sequences) and graphs. The first is to use
a dedicated convolution kernel, (e.g. ``FHTConv1d`` for sequences
or ``GraphRBF`` for graphs), e.g.:::

  from xGPR import xGPRegression
  my_model = xGPRegression(num_rffs = 2048,
                        variance_rffs = 512, kernel_choice = "FHTConv1d",
                        device = "gpu", kernel_settings =
                        {"conv_width":9}, verbose = True)

Everything else remains unchanged, you just need to ensure the dataset
you supply contains 3d arrays (otherwise a ValueError is raised). For
details on available convolution kernels, see the
Available Kernels section on the main page.

Another option is to use a feature extractor (aka "static layer"),
which is itself a kind of kernel, then feed the output of this feature
extractor into an ``RBF`` kernel. To do this, you'll 
need to create a "static layer" object and run your training
dataset through it. This static layer object will now become
part of your pipeline, and when making predictions you'll need to run any
array for which you want predictions through the "static layer"
as well.

We'll illustrate the properties of static layers / feature extractors
using the FastConv1d example, which is a type of kernel for sequences
and time series:

.. autoclass:: xGPR.FastConv1d
   :special-members: __init__
   :members: conv1d_pretrain_feat_extract, conv1d_x_feat_extract
