"""Checks that non-zero-padded variable length sequences can
be input to an FHTConv1d kernel with no errors."""
import sys

import unittest
import numpy as np

try:
    import cupy as cp
except:
    pass

#TODO: Get rid of this path modification
sys.path.append("..")
from utils.model_constructor import get_models


#A list of the kernels to be tested.
variable_length_kernels = ["FHTConv1d"]


class TestVariableLengthSeqs(unittest.TestCase):
    """Tests for errors when running non-zero-padded variable
    length sequences through a graph or sequence kernel."""

    def test_for_length_errors(self):
        """Tests the FHT-based FHTConv1d kernel."""
        rng = np.random.default_rng(123)
        block1 = rng.uniform(size=(100,20,21))
        block2 = rng.uniform(size=(100,10,21))
        dud_block = rng.uniform(size=(100,2,21))
        for kernel in variable_length_kernels:
            outcomes = run_kernel_specific_test(kernel, block1, block2,
                    dud_block)
            for outcome in outcomes:
                self.assertTrue(outcome)


def run_kernel_specific_test(kernel, block1, block2, dud_block):
    """Checks the specified kernel using both cpu and (if available)
    gpu on two valid input blocks (block1 and block2) and an invalid
    input block. Returns True for each test that passed and False
    for each that failed. This is a simple does it work or does it
    raise an exception test -- the correctness of the FHT operations
    is tested under the other tests in this folder."""
    models = get_models(kernel, block1.shape,
                        training_rffs = 1024, fitting_rffs = 1024,
                        conv_ard_kernel = False)
    outcomes = []
    for (model, device) in zip(models, ["cpu", "gpu"]):
        if model is None:
            continue
        if device == "gpu":
            block1 = cp.asarray(block1)
            block2 = cp.asarray(block2)
            dud_block = cp.asarray(dud_block)
        try:
            _ = model.kernel.transform_x(block1)
            outcomes.append(True)
        except:
            outcomes.append(False)
        try:
            _ = model.kernel.transform_x(block2)
            outcomes.append(True)
        except:
            outcomes.append(False)
        try:
            _ = model.kernel.transform_x(dud_block)
            outcomes.append(False)
        except:
            outcomes.append(True)
    return outcomes



if __name__ == "__main__":
    unittest.main()