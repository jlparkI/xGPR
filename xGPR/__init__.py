#Version number. Updated if generating a new release.
#Otherwise, do not change.
__version__ = "0.1.3.0"

#Key imports.
from .xGP_Regression import xGPRegression
from .data_handling.dataset_builder import build_online_dataset
from .data_handling.dataset_builder import build_offline_fixed_vector_dataset
from .data_handling.dataset_builder import build_offline_sequence_dataset
from .tuning_toolkit.bayesian_fitting_optimizer import BayesianFittingOptimizer
from .tuning_toolkit.direct_fitting_optimizer import DirectFittingOptimizer

from .kernel_xpca import KernelxPCA
from .kernel_fgen import KernelFGen
