"""Magic numbers for use by model classes."""
MAX_VARIANCE_RFFS = 4096
MAX_CLOSED_FORM_RFFS = 8192

#The following are default settings for kernels that have
#special parameters.
DEFAULT_KERNEL_SPEC_PARMS = {"matern_nu":5/2, "intercept":True,
        "averaging":"none"}

#The following is used if an error is encountered during
#matrix decomposition.
DEFAULT_SCORE_IF_PROBLEM = 1e40

#Default settings for NMLL approximation.
default_nmll_params = {"max_rank":1024, "preconditioner_mode":"srht_2",
        "nsamples":25, "nmll_iter":500, "nmll_tol":1e-6}

#Default max rank for NMLL approximation.
LARGEST_NMLL_MAX_RANK = 3000
#Default min rank for NMLL approximation.
SMALLEST_NMLL_MAX_RANK = 512
