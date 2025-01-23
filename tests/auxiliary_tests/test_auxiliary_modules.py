"""Tests the auxiliary tools (kernel pca, rf gen for kernel kmeans)
for obvious failures. This doesn't do any performance tests since
both are just wrappers on functionality used elsewhere; it
just ensures that they are functional."""
import sys
import os
import unittest

import numpy as np

from xGPR import KernelFGen

#TODO: Get rid of this path alteration
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.build_test_dataset import build_test_dataset
from utils.model_constructor import get_models
from utils.evaluate_model import evaluate_model


class CheckAuxiliaryFunctions(unittest.TestCase):
    """Checks whether the feature gen for
    kernel k-means is functioning."""

    def test_kernel_fgen(self):
        """Test the kernel fgen functionality."""
        online_sdata, _ = build_test_dataset(conv_kernel = False)
        online_cdata, _ = build_test_dataset(conv_kernel = True)

        #Ensure no exception if building the tool for standard datasets with standard
        #kernels and doing the usual things.
        raised = False
        kpca = KernelFGen(num_rffs = 500,
                            kernel_choice = "RBF", hyperparams = np.array([1.0]),
                            num_features = online_sdata.get_xdim()[-1],
                            random_seed = 123)
        xtrans = kpca.predict(online_sdata._xdata)
        self.assertFalse(raised, 'Does not work with RBF kernel.')
        self.assertTrue(xtrans.shape[1] == 500)

        raised = False
        try:
            kpca = KernelFGen(num_rffs = 500,
                            kernel_choice = "GraphRBF", hyperparams = np.array([1.0]),
                            num_features = online_cdata.get_xdim()[-1],
                            random_seed = 123)
            xtrans = kpca.predict(online_cdata._xdata, online_cdata._sequence_lengths)
        except:
            raised = True
        self.assertFalse(raised, 'Does not work with graph kernel.')
        self.assertTrue(xtrans.shape[1] == 500)

if __name__ == "__main__":
    unittest.main()
