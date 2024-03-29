#Version number. Updated if generating a new release.
#Otherwise, do not change.
__version__ = "0.3.2"

#Key imports.
from .xgp_regression import xGPRegression
from .xgp_classification import xGPDiscriminant
from .data_handling.dataset_builder import build_regression_dataset
from .data_handling.dataset_builder import build_classification_dataset

from .kernel_fgen import KernelFGen

from .static_layers import FastConv1d
