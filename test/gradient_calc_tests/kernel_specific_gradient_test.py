"""Provides a function for comparing an exact gradient with a
numerical gradient shared and used by all the kernel-specific
tests."""
import sys

import numpy as np
try:
    import cupy as cp
except:
    pass
from scipy.optimize import approx_fprime

#TODO: Get rid of this path modification
sys.path.append("..")
from utils.build_test_dataset import build_test_dataset
from utils.model_constructor import get_models


RANDOM_STATE = 123


def run_kernelspecific_test(kernel_choice, conv_kernel = False):
    """Compares a numerical gradient with an exact gradient using
    generic hyperparameters and generic kernel settings. Also compare
    with the minibatch gradient, which is calculated a little
    differently."""
    online_data, _ = build_test_dataset(conv_kernel)
    xdata, ydata, _ = online_data.get_next_minibatch(2000)
    cpu_mod, gpu_mod = get_models(kernel_choice, online_data.get_xdim())

    eps = np.sqrt(np.finfo(np.float32).eps)

    params = np.log(np.full(cpu_mod.get_hyperparams().shape, 0.5))

    cpu_cost, cpu_grad = cpu_mod.exact_nmll_gradient(params, online_data)
    cpu_mini_grad = cpu_mod.minibatch_nmll_gradient(params, xdata, ydata)
    singlepoint_cost = cpu_mod.exact_nmll(params, online_data)
    num_grad = approx_fprime(params, cpu_mod.exact_nmll,
                            eps, online_data)


    print(f"Analytic gradient, cpu:  {cpu_grad}")
    print(f"Minibatch gradient, cpu:  {cpu_mini_grad}")
    print(f"Numerical gradient, cpu: {num_grad}")
    print(f"Singlepoint cost, cpu:   {singlepoint_cost}")

    cpu_gradcomp = 100 * np.max(np.abs(cpu_grad - num_grad) / np.abs(num_grad))
    cpu_gradcomp = cpu_gradcomp < 0.25
    cpu_costcomp = 100 * np.abs(cpu_cost - singlepoint_cost) / singlepoint_cost
    cpu_costcomp = cpu_costcomp < 0.25
    cpu_mini_comp = 100 * np.max(np.abs(cpu_mini_grad - num_grad) / num_grad)
    cpu_mini_comp = cpu_mini_comp < 0.25

    if gpu_mod is not None:
        online_data.device = "gpu"
        gpu_cost, gpu_grad = gpu_mod.exact_nmll_gradient(params, online_data)
        xdata, ydata = cp.asarray(xdata), cp.asarray(ydata)

        gpu_mini_grad = gpu_mod.minibatch_nmll_gradient(params, xdata, ydata)
        singlepoint_cost = gpu_mod.exact_nmll(params, online_data)
        print(f"Analytic gradient, gpu:  {gpu_grad}")
        print(f"Minibatch gradient, gpu:  {gpu_mini_grad}")
        print(f"Singlepoint cost, gpu:   {singlepoint_cost}")

        gpu_gradcomp = 100 * np.max(np.abs(gpu_grad - cpu_grad) / cpu_grad)
        gpu_gradcomp = gpu_gradcomp < 0.25
        gpu_costcomp = 100 * np.abs(gpu_cost - singlepoint_cost) / singlepoint_cost
        gpu_costcomp = gpu_costcomp < 0.25
        gpu_mini_comp = 100 * np.max(np.abs(gpu_mini_grad - cpu_grad) / cpu_grad)
        gpu_mini_comp = gpu_mini_comp < 0.25
    else:
        gpu_gradcomp, gpu_costcomp = True, True
        gpu_mini_comp = True
    return cpu_gradcomp, cpu_costcomp, gpu_gradcomp, gpu_costcomp, \
            cpu_mini_comp, gpu_mini_comp
