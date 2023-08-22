"""Tests the auxiliary tools (kernel pca, rf gen for kernel kmeans)
for obvious failures. This doesn't do any performance tests since
both are just wrappers on functionality used elsewhere; it
just ensures that they are functional. (If we do implement
k-means, add performance tests for k-means here)."""
import sys
import unittest

import numpy as np

from xGPR import KernelxPCA
from xGPR import KernelFGen

#TODO: Get rid of this path alteration
sys.path.append("..")
from utils.build_test_dataset import build_test_dataset
from utils.model_constructor import get_models
from utils.evaluate_model import evaluate_model


class CheckAuxiliaryFunctions(unittest.TestCase):
    """Checks whether the auxiliary kPCA and feature gen for
    kernel k-means are functioning."""

    def test_kpca(self):
        """Test the approximate kernel pca functionality."""
        online_sdata, _ = build_test_dataset(conv_kernel = False)
        online_cdata, _ = build_test_dataset(conv_kernel = True)

        #Make sure that we can't build a kPCA with more components than RFFs
        #(we shouldn't be able to do that...)
        raised = False
        try:
            kpca = KernelxPCA(num_rffs = 500, n_components = 1000,
                            kernel_choice = "RBF", hyperparams = np.array([1.0]),
                            dataset = online_sdata, random_seed = 123)
        except:
            raised = True
        self.assertTrue(raised)

        #Ensure no exception if building kPCAs for standard datasets with standard
        #kernels and doing the usual things.
        raised = False
        try:
            kpca = KernelxPCA(num_rffs = 500, n_components = 2,
                            kernel_choice = "RBF", hyperparams = np.array([1.0]),
                            dataset = online_sdata, random_seed = 123)
            xtrans = kpca.predict(online_sdata.get_xdata())
        except:
            raised = True
        self.assertFalse(raised, 'Does not work with RBF kernel.')
        self.assertTrue(xtrans.shape[1] == 2)

        raised = False
        try:
            kpca = KernelxPCA(num_rffs = 500, n_components = 2,
                            kernel_choice = "GraphRBF", hyperparams = np.array([1.0]),
                            dataset = online_cdata, random_seed = 123)
            xtrans = kpca.predict(online_cdata.get_xdata())
        except:
            raised = True
        self.assertFalse(raised, 'Does not work with graph kernel.')
        self.assertTrue(xtrans.shape[1] == 2)


    def test_kernel_fgen(self):
        """Test the kernel fgen functionality."""
        online_sdata, _ = build_test_dataset(conv_kernel = False)
        online_cdata, _ = build_test_dataset(conv_kernel = True)

        #Ensure no exception if building the tool for standard datasets with standard
        #kernels and doing the usual things.
        raised = False
        try:
            kpca = KernelFGen(num_rffs = 500,
                            kernel_choice = "RBF", hyperparams = np.array([1.0]),
                            dataset = online_sdata, random_seed = 123)
            xtrans = kpca.predict(online_sdata.get_xdata())
        except:
            raised = True
        self.assertFalse(raised, 'Does not work with RBF kernel.')
        self.assertTrue(xtrans.shape[1] == 500)

        raised = False
        try:
            kpca = KernelFGen(num_rffs = 500,
                            kernel_choice = "GraphRBF", hyperparams = np.array([1.0]),
                            dataset = online_cdata, random_seed = 123)
            xtrans = kpca.predict(online_cdata.get_xdata())
        except:
            raised = True
        self.assertFalse(raised, 'Does not work with graph kernel.')
        self.assertTrue(xtrans.shape[1] == 500)

if __name__ == "__main__":
    unittest.main()
