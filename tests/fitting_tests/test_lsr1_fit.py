"""Tests fitting with an experimental L-SR1 method
for classification."""
import sys
import os
import unittest

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.build_test_dataset import build_test_dataset
from utils.build_classification_dataset import build_discriminant_traintest_split
from utils.model_constructor import get_models, get_discriminant_models
from utils.evaluate_model import evaluate_model

#A set of hyperparameters known to work well for our testing dataset
#that we can use as a default.
HPARAM = np.array([np.log(np.sqrt(0.0767)),  np.log(0.358)])
DISCRIM_HPARAM = np.array([0., -0.75])

NUM_RFFS = 4100


class CheckLSR1Fit(unittest.TestCase):
    """Tests the L-SR1 algorithm."""


    def test_preconditioned_discriminant_cg(self):
        """Test using L-SR1."""
        _, offline_data = build_discriminant_traintest_split()
        cpu_mod, gpu_mod = get_discriminant_models("RBF", offline_data,
                num_rffs = NUM_RFFS, model_type = "logistic")

        cpu_mod.set_hyperparams(DISCRIM_HPARAM, offline_data)
        niter, _ = cpu_mod.fit(offline_data,
                max_iter = 500, run_diagnostics = True, mode = "lsr1")
        print(f"niter: {niter}")
        self.assertTrue(niter < 200)

        preds, ylabels = [], []

        for xdata, ydata, _ in offline_data.get_chunked_data():
            preds.append(np.argmax(cpu_mod.predict(xdata), axis=1))
            ylabels.append(ydata)

        if gpu_mod is not None:
            gpu_mod.set_hyperparams(DISCRIM_HPARAM, offline_data)
            niter, _ = gpu_mod.fit(offline_data,
                max_iter = 500, run_diagnostics = True, mode = "lsr1")
            print(f"Discriminant classifier, niter: {niter}")
            self.assertTrue(niter < 200)



if __name__ == "__main__":
    unittest.main()
