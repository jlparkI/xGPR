"""Tests the classification tuning algorithms which
use a validation set to ensure they can achieve
100% accuracy on an easy dataset."""
import sys
import os
import unittest

import numpy as np

from xGPR import xGPDiscriminant, tune_classifier_optuna, tune_classifier_powell
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.build_classification_dataset import build_discriminant_traintest_split
from utils.model_constructor import get_discriminant_models


class CheckClassifierValidationSetTuning(unittest.TestCase):
    """Tests tuning on the validation set for the classifier."""

    def test_optuna_classifier_tuning(self):
        """Test the optuna tuning routine."""
        train_data, test_data = build_discriminant_traintest_split()
        for kernel in ["Linear", "RBF"]:
            models = [m for m in get_discriminant_models(kernel,
                train_data) if m is not None]

            for mod in models:
                mod.verbose = False
                _, best_score, _ = tune_classifier_optuna(train_data,
                        test_data, mod, fit_mode="exact",
                        eval_metric="accuracy", max_iter=50)
                self.assertTrue(best_score > 0.97)


    def test_powell_classifier_tuning(self):
        """Test the optuna tuning routine."""
        train_data, test_data = build_discriminant_traintest_split()
        for kernel in ["RBF"]:
            models = [m for m in get_discriminant_models(kernel,
                train_data) if m is not None]

            for mod in models:
                mod.verbose = False
                _, best_score, _ = tune_classifier_powell(train_data,
                        test_data, mod, fit_mode="exact",
                        eval_metric="accuracy", n_restarts=4)
                self.assertTrue(best_score > 0.97)


if __name__ == "__main__":
    unittest.main()
