"""Tests the static layer construction to ensure they
initialize and can transform data."""
import sys
import unittest
import os

from xGPR.static_layers import FastConv1d

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.build_test_dataset import build_test_dataset


RANDOM_SEED = 123


class CheckStatLayerConstruction(unittest.TestCase):
    """Tests construction of static layer objects."""

    def test_static_layer_builders(self):
        """Test static layer construction and basic functions."""
        test_online_dataset, _ = build_test_dataset(conv_kernel = True,
            xsuffix = "testxvalues.npy", ysuffix = "testyvalues.npy")
        train_online_dataset, _ = build_test_dataset(conv_kernel = True)

        conv_statlayer = FastConv1d(test_online_dataset.get_xdim()[2],
                device = "cpu", random_seed = RANDOM_SEED, conv_width = 3,
                num_features = 512)

        xchunks = list(train_online_dataset.get_chunked_x_data())
        x_trans = conv_statlayer.predict(xchunks[0][0], xchunks[0][1])
        self.assertTrue(x_trans.shape[1] == 512)
        self.assertTrue(xchunks[0][0].shape[0] == x_trans.shape[0])


if __name__ == "__main__":
    unittest.main()
