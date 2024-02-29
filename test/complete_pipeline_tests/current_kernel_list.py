"""This file lists key currently implemented
kernels, together with the performance (the spearman's r)
expected from each on the test set included with the
module. Add to this when a new kernel is added to
ensure that kernel is included; you will need to
decide what is 'acceptable performance', likely
based on an initial experiment. Note that kernels
which are essentially modified versions of RBF
kernels are excluded (e.g. Conv1dMatern does
not need to be tested since Conv1dRBF is)."""

#Each dictionary value contains 1) whether this
#is a convolution kernel and 2) the expected
#minimum performance.
IMPLEMENTED_KERNELS = {
        "Conv1dRBF":(True,0.58),"RBF":(False, 0.58),
        "Matern":(False, 0.55),
        "Linear":(False, 0.53),
        "RBFLinear":(False,0.55),
        "MiniARD":(False, 0.64),
        "Conv1dRBF":(True, 0.58),
        "GraphRBF":(True, 0.38),
        }
