"""Stores a dict of all kernel names mapping each to a
class. Should be updated any time a new kernel is
added / any time one is removed."""
from .basic_kernels.matern import Matern
from .basic_kernels.rbf import RBF
from .basic_kernels.linear import Linear
from .basic_kernels.polynomial import Polynomial

from .convolution_kernels.fht_conv1d import FHTConv1d
from .convolution_kernels.graph_rbf import GraphRBF
from .convolution_kernels.graph_polysum import GraphPolySum

from .ARD_kernels.mini_ard import MiniARD


KERNEL_NAME_TO_CLASS = {"RBF":RBF,
        "Matern":Matern,
        "FHTConv1d":FHTConv1d,
        "GraphRBF":GraphRBF,
        "Linear":Linear,
        "Poly":Polynomial,
        "GraphPoly":GraphPolySum,
        "MiniARD":MiniARD}

# A list of kernels that require 3d arrays as input. This
# is used by kernel_fgen for generating random features
# outside of a regressor / classifier.
ARR_3D_KERNELS = ("GraphRBF", "GraphPoly", "FHTConv1d")
