"""Stores a dict of all kernel names mapping each to a
class. Should be updated any time a new kernel is
added / any time one is removed."""
from .basic_kernels.matern import Matern
from .basic_kernels.rbf import RBF
from .basic_kernels.linear import Linear
from .basic_kernels.cauchy import Cauchy
from .basic_kernels.rbf_linear import RBFLinear

from .convolution_kernels.conv1d_rbf import Conv1dRBF
from .convolution_kernels.conv1d_matern import Conv1dMatern
from .convolution_kernels.conv1d_cauchy import Conv1dCauchy
from .convolution_kernels.l2_conv1d import Conv1dTwoLayer

from .convolution_kernels.graph_rbf import GraphRBF
from .convolution_kernels.graph_matern import GraphMatern
from .convolution_kernels.graph_cauchy import GraphCauchy

from .ARD_kernels.mini_ard import MiniARD


KERNEL_NAME_TO_CLASS = {"RBF":RBF,
        "Matern":Matern,
        "Cauchy":Cauchy,
        "Conv1dRBF":Conv1dRBF,
        "Conv1dMatern":Conv1dMatern,
        "Conv1dCauchy":Conv1dCauchy,
        "Conv1dTwoLayer":Conv1dTwoLayer,
        "GraphRBF":GraphRBF,
        "GraphMatern":GraphMatern,
        "GraphCauchy":GraphCauchy,
        "Linear":Linear,
        "RBFLinear":RBFLinear,
        "MiniARD":MiniARD}

# A list of kernels that require 3d arrays as input. This
# is used by kernel_fgen for generating random features
# outside of a regressor / classifier.
ARR_3D_KERNELS = {"GraphRBF", "Conv1dRBF", "Conv1dMatern",
        "GraphMatern", "GraphCauchy", "Conv1dCauchy",
        "Conv1dTwoLayer"}
