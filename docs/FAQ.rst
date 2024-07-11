FAQ
====

#. **xGPR uses random features. How much does performance fluctuate if I
   choose a different random seed?**

   Not much at all, as long as you're using a decent number of RFFs. See
   our original paper. Performance is quite reproducible with different
   random seeds. It's still nonetheless good practice to save the random
   seed you use (as always).

#. **Training with multiple GPUs -- is this possible?**

   The xGPR algorithm can be easily parallelized, and this feature
   is planned for a near-future version of xGPR.

#. **I have multiple GPUs. When I set "model.device = 'cuda'", which
   gpu is used?**
   Whichever one is currently active. You can determine the currently
   active cuda device by setting the environment variable,
   e.g. ``export CUDA_VISIBLE_DEVICES=0`` so that gpu 0 is used.
   This isn't ideal -- we'll add capability to allow setting a
   specific device in xGPR rather than via an environment variable
   in future.

#. **Why doesn't xGPR have (insert my favorite kernel here)**
   
   We might be interested in adding other kernels to xGPR, but with a few
   caveats.
   
   * It has to be a kernel for which the random features approach can be 
     implemented in an at least-somewhat straightforward way. Random features
     is much faster and more scalable than stochastic variational GPs.
     So much so, in fact, that for our purposes it's not really worth
     implementing a kernel that can't use random features.

   * We prefer kernels that are applicable to a range of problems rather
     than a single specific problem.

   * We're less enthusiastic about kernels that have a large number of
     hyperparameters that need to be optimized. In our experience these
     are harder to work with (with some exceptions).

   If you have a kernel you'd really like to see -- see the Contact page,
   we'd love to hear more. For more background, see :doc:`When should I use 
   xGPR?</purpose>`.
