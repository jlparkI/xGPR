#Version number. Updated if generating a new release.
#Otherwise, do not change.
__version__ = "0.4.7"

#Key imports.
from .xgp_regression import xGPRegression
from .data_handling.data_handling_baseclass import DatasetBaseclass
from .data_handling.dataset_builder import build_regression_dataset
from .data_handling.dataset_builder import build_classification_dataset

from .kernel_fgen import KernelFGen

from .static_layers import FastConv1d
