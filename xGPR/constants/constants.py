"""Magic numbers for use by model classes."""
#The following are used by model classes.
MAX_VARIANCE_RFFS = 4096
MAX_CLOSED_FORM_RFFS = 8192

#The following are default settings for kernels that have
#special parameters.
DEFAULT_KERNEL_SPEC_PARMS = {"matern_nu":5/2, "conv_width":9,
        "polydegree":2}

#The following is used if an error is encountered during
#matrix decomposition.
DEFAULT_SCORE_IF_PROBLEM = 1e40
