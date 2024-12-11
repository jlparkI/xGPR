"""Checks that non-zero-padded variable length sequences can
be input to an FHTConv1d kernel with no errors."""
import sys
import os
import unittest
import numpy as np

try:
    import cupy as cp
except:
    pass

from xGPR import build_regression_dataset
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.model_constructor import get_models


#A list of the kernels to be tested.
variable_length_kernels = ["Conv1dRBF"]


class TestVariableLengthSeqs(unittest.TestCase):
    """Tests for errors when running non-zero-padded variable
    length sequences through a graph or sequence kernel."""

    def test_for_length_errors(self):
        """Tests the Conv1dRBF kernel."""
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
    cuda on two valid input blocks (block1 and block2) and an invalid
    input block. Returns True for each test that passed and False
    for each that failed. This is a simple does it work or does it
    raise an exception test -- the correctness of the FHT operations
    is tested under the other tests in this folder."""
    sequence_lengths = np.full(block1.shape[0],
                block1.shape[1]).astype(np.int32)
    online_dataset = build_regression_dataset(block1,
            np.zeros((block1.shape[0])), sequence_lengths)
    models = get_models(kernel, online_dataset,
                        num_rffs = 1024,
                        conv_ard_kernel = False)
    outcomes = []
    for (model, device) in zip(models, ["cpu", "cuda"]):
        if model is None:
            continue

        sequence_lengths = np.full(block1.shape[0],
                block1.shape[1]).astype(np.int32)
        try:
            _ = model.kernel.transform_x(block1, sequence_lengths)
            outcomes.append(True)
        except:
            outcomes.append(False)

        try:
            sequence_lengths = np.full(block2.shape[0],
                    block2.shape[1]).astype(np.int32)
            _ = model.kernel.transform_x(block2, sequence_lengths)
            outcomes.append(True)
        except:
            outcomes.append(False)

        try:
            sequence_lengths = np.full(dud_block.shape[0],
                    dud_block.shape[1]).astype(np.int32)
            _ = model.kernel.transform_x(dud_block, sequence_lengths)
            outcomes.append(False)
        except:
           outcomes.append(True)

        try:
            sequence_lengths = np.full(dud_block.shape[0],
                    dud_block.shape[1]).astype(np.int32)
            _ = model.kernel.transform_x(block2, sequence_lengths)
            outcomes.append(False)
        except:
           outcomes.append(True)

        try:
            sequence_lengths = np.full(dud_block.shape[0], 21).astype(np.int32)
            _ = model.kernel.transform_x(block2, sequence_lengths)
            outcomes.append(False)
        except:
           outcomes.append(True)

    return outcomes



if __name__ == "__main__":
    unittest.main()
