"""Tests using both exact fitting and preconditioned CG with
minimal bayes tuning to ensure that we can achieve expected
performance for all currently implemented kernels. This is an
'all-in-one' workflow test, if it fails, run fitting tests,
tuning tests, preconditioner tests and fht operations tests
as appopriate to determine which component is failing."""
import unittest
import sys
import os

from current_kernel_list import IMPLEMENTED_KERNELS

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.build_test_dataset import build_test_dataset
from utils.model_constructor import get_models
from utils.evaluate_model import evaluate_model




class CheckPipeline(unittest.TestCase):
    """An all in one pipeline test."""



    def test_fit_cpu(self):
        """Test on cpu."""
        print("Now running CPU tests. Some of these (primarily any "
                "involving ARD kernels) may take a minute.")
        for kernel_name, (is_conv, exp_score) in IMPLEMENTED_KERNELS.items():
            cg_score, exact_score = fit_and_evaluate(kernel_name, is_conv,
                conv_width = 3, get_var = True, device="cpu")
            self.assertTrue(cg_score > exp_score)
            self.assertTrue(exact_score > exp_score)

    def test_fit_gpu(self):
        """Test on gpu."""
        print("Now running GPU tests.")
        for kernel_name, (is_conv, exp_score) in IMPLEMENTED_KERNELS.items():
            cg_score, exact_score = fit_and_evaluate(kernel_name, is_conv,
                conv_width = 3, get_var = True, device="gpu")
            self.assertTrue(cg_score > exp_score)
            self.assertTrue(exact_score > exp_score)


def fit_and_evaluate(kernel, conv_kernel, conv_width = 3,
            get_var = True, conv_ard_kernel = False,
            training_rffs = 512, cg_fitting_rffs = 8192,
            exact_fitting_rffs = 2048, device = "gpu"):
    """Test on a specified device using preconditioned CG and exact."""
    _, train_dataset = build_test_dataset(conv_kernel = conv_kernel)
    cpu_mod, gpu_mod = get_models(kernel, train_dataset, conv_width,
                            conv_ard_kernel = conv_ard_kernel,
                            num_rffs = training_rffs)
    if device == "gpu":
        if gpu_mod is None:
            #If GPU not available, return immediately.
            return None, None
        else:
            model = gpu_mod
    else:
        model = cpu_mod

    model.verbose = False

    test_dataset, _ = build_test_dataset(conv_kernel = conv_kernel,
            xsuffix = "testxvalues.npy", ysuffix = "testyvalues.npy")
    if kernel == "MiniARD":
        model.tune_hyperparams(train_dataset, n_restarts = 1, tol=1e-2,
            tuning_method = "L-BFGS-B")
    else:
        model.tune_hyperparams_crude(train_dataset)


    print(f"Hyperparams, cpu, {kernel}: {model.get_hyperparams()}")
    model.num_rffs = cg_fitting_rffs

    model.fit(train_dataset, max_iter = 500, tol = 1e-6,  mode = "cg")
    cg_score = evaluate_model(model, train_dataset, test_dataset,
            get_var)

    print(f"CG score, {device}, {kernel}: {cg_score}")

    model.num_rffs = exact_fitting_rffs

    model.fit(train_dataset, mode = "exact")
    exact_score = evaluate_model(model, train_dataset, test_dataset,
            get_var)
    print(f"Exact score, {device}, {kernel}: {exact_score}")

    return cg_score, exact_score




if __name__ == "__main__":
    unittest.main()
