"""Tests the static layer construction to ensure they
initialize and can transform data."""
import sys
import unittest
import os

from xGPR.static_layers import FastConv1d

#TODO: Get rid of this path alteration
sys.path.append("..")
from utils.build_test_dataset import build_test_dataset


RANDOM_SEED = 123


class CheckStatLayerConstruction(unittest.TestCase):
    """Tests construction of static layer objects."""

    def test_static_layer_builders(self):
        """Test static layer construction and basic functions."""
        test_online_dataset, test_offline_dataset = build_test_dataset(conv_kernel = True,
            xsuffix = "testxvalues.npy", ysuffix = "testyvalues.npy")
        train_online_dataset, _ = build_test_dataset(conv_kernel = True)

        conv_statlayer = FastConv1d(test_online_dataset.get_xdim()[2],
                device = "cpu", random_seed = RANDOM_SEED, conv_width = 3,
                num_features = 512)
        conv_dset = conv_statlayer.conv1d_pretrain_feat_extract(test_offline_dataset,
                os.getcwd())

        xchunks = list(train_online_dataset.get_chunked_x_data())
        x_trans = conv_statlayer.conv1d_feat_extract(xchunks[0][0], xchunks[0][1])
        self.assertTrue(x_trans.shape[1] == 512)
        self.assertTrue(xchunks[0][0].shape[0] == x_trans.shape[0])

        for xfile in conv_dset.get_xfiles():
            os.remove(xfile)
        for yfile in conv_dset.get_yfiles():
            os.remove(yfile)


if __name__ == "__main__":
    unittest.main()
