"""Stores a dict of all kernel names mapping each to a
class. Should be updated any time a new kernel is
added / any time one is removed."""
from .basic_kernels.matern import Matern
from .basic_kernels.rbf import RBF
from .basic_kernels.linear import Linear
from .basic_kernels.polynomial import Polynomial

from .convolution_kernels.fht_conv1d import FHTConv1d
from .convolution_kernels.mm_conv1d import MMConv1d
from .convolution_kernels.graph_fht_conv1d import GraphFHTConv1d
from .convolution_kernels.graph_polysum import GraphPolySum

KERNEL_NAME_TO_CLASS = {"RBF":RBF,
        "Matern":Matern,
        "Conv1d":MMConv1d,
        "FHTConv1d":FHTConv1d,
        "GraphConv1d":GraphFHTConv1d,
        "Linear":Linear,
        "Poly":Polynomial,
        "GraphPoly":GraphPolySum}
