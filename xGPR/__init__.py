#Version number. Updated if generating a new release.
#Otherwise, do not change.
__version__ = "0.1.3.2"

#Key imports.
from .xGP_Regression import xGPRegression
from .xGP_Classification import xGPClassifier
from .data_handling.dataset_builder import build_regression_dataset
from .data_handling.dataset_builder import build_classification_dataset

from .kernel_fgen import KernelFGen

from .static_layers import FastConv1d
