"""Provides a function for comparing an exact gradient with a
numerical gradient shared and used by all the kernel-specific
tests."""
import sys
import os
import numpy as np
try:
    import cupy as cp
except:
    pass
from scipy.optimize import approx_fprime

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.build_test_dataset import build_test_dataset
from utils.model_constructor import get_models


RANDOM_STATE = 123


def run_kernelspecific_test(kernel_choice, conv_kernel = False,
                training_rffs = 512, conv_ard_kernel = False,
                averaging = 'none'):
    """Compares a numerical gradient with an exact gradient using
    generic hyperparameters and generic kernel settings."""
    online_data, _ = build_test_dataset(conv_kernel)
    xdata, ydata = online_data._xdata[:2000,...], online_data._ydata[:2000]
    cpu_mod, gpu_mod = get_models(kernel_choice, online_data,
                        num_rffs = training_rffs, conv_ard_kernel = conv_ard_kernel,
                        averaging = averaging)

    eps = np.sqrt(np.finfo(np.float32).eps)

    params = np.log(np.full(cpu_mod.get_hyperparams().shape, 0.1))

    cpu_cost, cpu_grad = cpu_mod.exact_nmll_gradient(params, online_data)
    singlepoint_cost = cpu_mod.exact_nmll(params, online_data)
    num_grad = approx_fprime(params, cpu_mod.exact_nmll,
                            eps, online_data)


    print(f"Kernel: {kernel_choice}")
    print(f"Analytic gradient, cpu:  {cpu_grad}")
    print(f"Numerical gradient, cpu: {num_grad}")
    print(f"Singlepoint cost, cpu:   {singlepoint_cost}")

    cpu_gradcomp = 100 * np.max(np.abs(cpu_grad - num_grad) / np.abs(cpu_grad))
    cpu_gradcomp = cpu_gradcomp < 0.5 or np.max(np.abs(cpu_grad - num_grad)) < 0.1
    cpu_costcomp = 100 * np.abs(cpu_cost - singlepoint_cost) / np.abs(singlepoint_cost)
    cpu_costcomp = cpu_costcomp < 0.25

    if gpu_mod is not None:
        gpu_cost, gpu_grad = gpu_mod.exact_nmll_gradient(params, online_data)
        xdata, ydata = cp.asarray(xdata), cp.asarray(ydata)

        singlepoint_cost = gpu_mod.exact_nmll(params, online_data)
        print(f"Analytic gradient, gpu:  {gpu_grad}")
        print(f"Singlepoint cost, gpu:   {singlepoint_cost}")

        gpu_gradcomp = 100 * np.max(np.abs(gpu_grad - cpu_grad) / np.abs(cpu_grad))
        gpu_gradcomp = gpu_gradcomp < 0.5 or np.max(np.abs(gpu_grad - num_grad)) < 0.1
        gpu_costcomp = 100 * np.abs(gpu_cost - singlepoint_cost) / np.abs(singlepoint_cost)
        gpu_costcomp = gpu_costcomp < 0.25
    else:
        gpu_gradcomp, gpu_costcomp = True, True
    return cpu_gradcomp, cpu_costcomp, gpu_gradcomp, gpu_costcomp
