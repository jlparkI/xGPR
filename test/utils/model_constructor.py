"""Builds xGPRegression models with generic kernel parameters for
use in other tests."""
import sys
import copy

from xGPR import xGPRegression as xGPReg

RANDOM_STATE = 123


def get_models(kernel_choice, dataset, conv_width = 3, num_rffs = 512,
        conv_ard_kernel = False, averaging = False):
    """Generates a CPU model and a GPU model with generic
    kernel settings."""
    if not conv_ard_kernel:
        split_pts = [21,42,63]
    else:
        split_pts = [8]

    cpu_mod = xGPReg(num_rffs = num_rffs, kernel_choice = kernel_choice,
            variance_rffs = 12, random_seed = RANDOM_STATE, device = "cpu",
            kernel_specific_params = {"matern_nu":5/2,
                            "conv_width":conv_width, "polydegree":2,
                            "split_points":split_pts, "order":2,
                            "intercept":True, "averaging":averaging})
    if "cupy" not in sys.modules:
        print("Cupy not installed -- skipping the CUDA test.")
        gpu_mod = None
    else:
        gpu_mod = copy.deepcopy(cpu_mod)
        gpu_mod.device = "gpu"
        gpu_mod.set_hyperparams(dataset = dataset)

    cpu_mod.set_hyperparams(dataset = dataset)
    return cpu_mod, gpu_mod
