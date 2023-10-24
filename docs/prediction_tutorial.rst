Making predictions & visualizing / clustering results
======================================================

Note that you don't need to rescale y-values -- xGPR always handles this
automatically (unless you set ``normalize_y`` to False when building the
dataset). If you rescaled your x-values, e.g. to zero mean and unit
variance, however, you'll need to take care of that piece yourself.

To make predictions:::

  predicted_mean, predicted_variance = my_model.predict(xtest, chunk_size = 2000,
                                           get_var = False)

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
