"""Tests exact and CG fitting for the discriminant classifier to ensure it yields
the expected performance on the simple Wine dataset."""
import sys
import unittest
import numpy as np
from sklearn.metrics import accuracy_score

#TODO: Get rid of this path alteration
sys.path.append("..")
from utils.build_classification_dataset import build_discriminant_traintest_split
from utils.model_constructor import get_discriminant_models

#A set of hyperparameters known to work well for our testing dataset
#that we can use as a default.
DISCRIM_HPARAM = np.array([0., -0.75])



RANDOM_STATE = 123
CG_RFFS = 4096
EXACT_RFFS = 1024


class CheckDiscriminantPipeline(unittest.TestCase):
    """An all in one pipeline test."""

    def test_discriminant_fit(self):
        """Test on both devices using preconditioned CG and exact."""
        train_data, test_data = build_discriminant_traintest_split()
        cpu_mod, gpu_mod = get_discriminant_models("RBF", train_data,
                num_rffs = CG_RFFS)

        for model in [cpu_mod, gpu_mod]:
            if model is None:
                continue

            model.verbose = False
            model.set_hyperparams(DISCRIM_HPARAM, train_data)
            model.fit(train_data, max_iter = 500,
                    tol = 1e-6,  mode = "cg")

            y_pred = model.predict(test_data.get_xdata())
            cg_score = accuracy_score(np.argmax(y_pred, axis=1), test_data.get_ydata())
            self.assertTrue(cg_score > 0.95)

            print(f"CG discriminant score, wine data: {cg_score}")

            model.num_rffs = EXACT_RFFS

            model.fit(train_data, mode = "exact")
            y_pred = model.predict(test_data.get_xdata())
            exact_score = accuracy_score(np.argmax(y_pred, axis=1), test_data.get_ydata())
            self.assertTrue(exact_score > 0.95)
            print(f"Exact discriminant score, wine data: {exact_score}")

if __name__ == "__main__":
    unittest.main()
