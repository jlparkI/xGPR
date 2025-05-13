# Version number. Updated if generating a new release.
# Otherwise, do not change.
__version__ = "0.4.8.5"

# Key imports.
from .xgp_regression import xGPRegression
from .xgp_classification import xGPDiscriminant
from .data_handling.data_handling_baseclass import DatasetBaseclass
from .data_handling.dataset_builder import build_regression_dataset
from .data_handling.dataset_builder import build_classification_dataset

from .kernel_fgen import KernelFGen

from .static_layers import FastConv1d

from .scoring_toolkit.validation_set_tuning import tune_classifier_powell
try:
    from .scoring_toolkit.validation_set_tuning import tune_classifier_optuna
    from .scoring_toolkit.validation_set_tuning import cv_tune_classifier_optuna
except:
    pass
