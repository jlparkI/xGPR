FAQ
====

#. **Why doesn't xGPR run on Windows (yet)?**
   
   xGPR uses pthreads for multithreading. There's a Windows port of pthreads
   that we need to include (it's a near-term priority...)

#. **xGPR uses random features. How much does performance fluctuate if I
   choose a different random seed?**

   Not much at all, as long as you're using a decent number of RFFs. See
   our original paper. Performance is quite reproducible with different
   random seeds. It's still nonetheless good practice to save the random
   seed you use (as always).

#. **Training with multiple GPUs -- is this possible?**

   The xGPR algorithm can be easily parallelized, and this feature
   is upcoming in a near-future version of xGPR.

#. **Why doesn't xGPR have (insert my favorite kernel here)**
   
   We're interested in adding more kernels to xGPR, but any kernel we add 
   must meet three requirements.
   
   * It has to be a kernel for which the random features approach can be 
     implemented in an at least-somewhat straightforward way. Random features
     is much faster and more scalable than stochastic variational GPs.
     So much so, in fact, that for our purposes it's not really worth
     implementing a kernel that can't use random features.

   * It should be something you can use for general machine learning problems,
     rather than something specific to a specific problem.

   If you have a kernel that meets all these criteria -- shoot us an email,
   we'd love to hear more. For more background, see :doc:`When should I use 
   xGPR?</purpose>`.
