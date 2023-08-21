CPU Training -- tricks and tips
===============================

We prefer training on GPU wherever possible;
it's much faster and therefore more painless. (Making
predictions, by contrast, while faster on GPU, is also
reasonable on CPU as well.) If you do have to train on CPU,
however, here are some tips to help you achieve reasonable
speed for your training process.

  * Always pretransform your data when fitting, or supply
    a ``pretransform_dir`` if using the ``tuning_toolkit``
    or tuning with approximate marginal likelihood.
    Generating random features once and saving them on disk
    at the beginning of fitting can actually be slower on
    GPU for some kernels, but for CPU it is nearly always
    faster.
  
  * Always build preconditioners using ``method = 'srht'``.
    While 'srht_2' yields a somewhat-better-quality preconditioner,
    it requires matrix multiplications which are slow on CPU
    while 'srht' does not.
  
  * Try to avoid using a ``crude`` tuning method with a large number
    of ``training_rffs`` (i.e. > 2048); this will be quite slow on CPU.
    Instead, find a preliminary set of "good starting point"
    hyperparameters using a small number of RFFs (e.g. 512 or 1024),
    and use this as a starting point for "fine" approximate marginal
    likelihood or validation-set based tuning.
