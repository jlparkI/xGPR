Making predictions & visualizing / clustering results
======================================================

Note that you don't need to rescale y-values -- xGPR always handles this
automatically. If you rescaled your x-values, e.g. to zero mean and unit
variance, however, you'll need to take care of that piece yourself.

To make predictions:::

  predicted_mean, predicted_variance = my_model.predict(xtest, chunk_size = 2000,
                                           get_var = True)

xtest in this case is a numpy array with the same ``shape[1]`` and (if a convolution
kernel) ``shape[2]`` as your original data. You don't need to build a Dataset when
making predictions, only when training. ``chunk_size`` controls the maximum
chunk for which predictions are generated at any one time to limit
memory consumption.

``predicted_mean`` is the actual prediction, while ``predicted_variance`` is how
confident the model is about its prediction (bigger values = less confident). This is
one of the great advantages of GPs -- the model will automatically tell you
when it doesn't trust its own prediction. The predicted variance will *always*
increase to very large values once you go to points that lie well outside
the region covered by the original training set. You may not
need the predicted variance, in which case you can set ``get_var`` to ``False``
and only the predicted_mean is returned. This will be faster so is recommended
any time you don't need the variance.

Note that if you used a static layer to preprocess your training set,
you must also preprocess future data through the same static layer
when making predictions, e.g.:::

  my_new_test_array = my_stat_layer.conv1d_x_feat_extract(my_xtest,
                         chunk_size = 2000)
  predicted_mean, predicted_variance = my_model.predict(my_new_test_array,
                           chunk_size = 2000, get_var = True)

Finally, while we prefer training on GPU, it may often be desirable to
perform inference on CPU. You can switch the model back and forth as follows:::

  my_model.device = "gpu"
  my_model.device = "cpu"

You can do the same for static layers as well.

Saving models
------------------------

Currently there is no built-in model saving feature in xGPR; we recommend using ``pickle`` or
``joblib`` to save a model to disk.

It's also a good idea to keep track of the best set of hyperparameters that
you find while tuning. Tuning is more expensive and more of a hassle than fitting, and
you can also use an existing set of hyperparameters as a starting point for "fine-tuning",
which can sometimes save you time if you want to re-tune once additional data becomes
available.


Clustering and visualizing data
---------------------------------

Just as a neural network generates an "embedded representation" of its
input, so too does a GP generate a "representation" of the input
datapoints -- in this case, the random features. You can cluster the
random features-representation of some data or use it to do PCA, and
this is the same in effect as doing kernel PCA or kernel-based clustering
(e.g. kernel-kmeans).

For example, here is a kernel PCA of the QM9 dataset (130,000 small
molecules) generated using the random features produced by a GraphConv1d
kernel, trained on one-hot encoded input data:

.. image:: images/kernel_pca.png
   :width: 350
   :alt: Kernel PCA


In this case, you can see that the label we want to predict correlates quite
well with the first principal component. This will not always be true, of course,
because using only two principal components is of course throwing away a lot
of the information contained in (in this case) the 16,000 random feature
representation.

If you want to generate random features for an input numpy array, you can
call:::

  my_model.transform_data(input_x, chunk_size = 2000)

This function is a generator, so it will loop over ``input_x`` and return
chunks of size ``(chunk_size, fitting_rffs)``. You can therefore use it
in a loop.

In general, if you are using a model with say 16,000 ``fitting_rffs``, generating
all of the random features for all of the datapoints you want to cluster and
then clustering them may be quite expensive. It is often better and more
straightforward to use the experimental kernel PCA tool provided with
xGPR. This tool will perform an approximate PCA on the random
features representation and supply you with the top *n* approximate
principal components. Here's how to use it:::

  from xGPR.visualization_tools.kernel_xpca import kernel_xPCA

  kernel_xpca = kernel_xPCA(my_dataset, my_model, n_components=2,
                            random_seed = 123)

  my_pca_transformed_data = kernel_xpca.predict(my_input_numpy_array,
                                           chunk_size = 2000)

See the prediction section above to understand the ramifications
of ``chunk_size``. The ``kernel_xpca`` object when created fits
an approximate PCA, and then when ``predict`` is called, returns the
transformed inputs projected onto the (approximate) top ``n_components``
eigenvectors, so that ``my_pca_transformed_data`` will have shape
``(my_input_numpy_array.shape[0], n_components)``. For generating
a kernel PCA plot, just use 2 components; for clustering,
100 - 400 is likely more appropriate. You can then cluster this
data with whatever tool you find appropriate.

Be aware that the ``visualization_tools`` module is still in active development.
It is not therefore recommended to use it in production at this time. It
should be used for exploratory data analysis.
