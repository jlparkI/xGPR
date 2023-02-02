"""Builds xGPRegression models with generic kernel parameters for
use in other tests."""
import sys
import copy

from xGPR.xGP_Regression import xGPRegression as xGPReg

RANDOM_STATE = 123


def get_models(kernel_choice, xdim, conv_width = 3, training_rffs = 512,
        fitting_rffs = 512):
    """Generates a CPU model and a GPU model with generic
    kernel settings."""
    cpu_mod = xGPReg(training_rffs = training_rffs, fitting_rffs = fitting_rffs,
                        kernel_choice = kernel_choice,
                        device = "cpu", double_precision_fht = False,
                        kernel_specific_params = {"matern_nu":5/2,
                            "conv_width":conv_width, "polydegree":2})
    if "cupy" not in sys.modules:
        print("Cupy not installed -- skipping the CUDA test.")
        gpu_mod = None
    else:
        gpu_mod = copy.deepcopy(cpu_mod)
        gpu_mod.device = "gpu"
        gpu_mod.kernel = gpu_mod._initialize_kernel(gpu_mod.kernel_choice, xdim,
                    gpu_mod.training_rffs, RANDOM_STATE)

    #We access a protected class member here because for testing purposes
    #we need to initialize a kernel without trying to tune or fit, which is
    #not something a user will ever need.
    cpu_mod.kernel = cpu_mod._initialize_kernel(cpu_mod.kernel_choice, xdim,
                    cpu_mod.training_rffs, RANDOM_STATE)
    return cpu_mod, gpu_mod
