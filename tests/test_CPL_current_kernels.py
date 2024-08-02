"""Tests using both exact fitting and preconditioned CG with
minimal bayes tuning to ensure that we can achieve expected
performance for all currently implemented kernels. This is an
'all-in-one' workflow test, if it fails, run fitting tests,
tuning tests, preconditioner tests and fht operations tests
as appopriate to determine which component is failing."""
import unittest
from utils.build_test_dataset import build_test_dataset
from utils.model_constructor import get_models
from utils.evaluate_model import evaluate_model


#Each dictionary value contains 1) whether this
#is a convolution kernel and 2) the expected
#minimum performance.
IMPLEMENTED_KERNELS = {
        "Conv1dRBF":(True,0.58),
        "RBF":(False, 0.58),
        "Matern":(False, 0.55),
        "Linear":(False, 0.53),
        "RBFLinear":(False,0.55),
        "MiniARD":(False, 0.64),
        "GraphRBF":(True, 0.38),
        }



class CheckPipeline(unittest.TestCase):
    """An all in one pipeline test."""



    def test_fit(self):
        """Test on both devices."""
        print("Now running CPU tests. Some of these (primarily any "
                "involving ARD kernels) may take a minute.")
        for kernel_name, (is_conv, exp_score) in IMPLEMENTED_KERNELS.items():
            _, train_dataset = build_test_dataset(conv_kernel = is_conv)
            cpu_mod, _ = get_models(kernel_name, train_dataset, 3,
                            num_rffs = 512)
            cpu_mod.verbose = False
            test_dataset, _ = build_test_dataset(conv_kernel = is_conv,
                    xsuffix = "testxvalues.npy", ysuffix = "testyvalues.npy")
            cg_score, exact_score = cpl_evaluate_model(cpu_mod, train_dataset,
                    test_dataset)
            self.assertTrue(cg_score > exp_score)
            self.assertTrue(exact_score > exp_score)

        for kernel_name, (is_conv, exp_score) in IMPLEMENTED_KERNELS.items():
            _, train_dataset = build_test_dataset(conv_kernel = is_conv)
            _, gpu_mod = get_models(kernel_name, train_dataset, 3,
                            num_rffs = 512)
            if gpu_mod is None:
                continue
            gpu_mod.verbose = False
            test_dataset, _ = build_test_dataset(conv_kernel = is_conv,
                    xsuffix = "testxvalues.npy", ysuffix = "testyvalues.npy")
            cg_score, exact_score = cpl_evaluate_model(gpu_mod, train_dataset,
                    test_dataset)
            self.assertTrue(cg_score > exp_score)
            self.assertTrue(exact_score > exp_score)


def cpl_evaluate_model(model, train_dataset, test_dataset):
    if model.kernel_choice == "MiniARD":
        model.tune_hyperparams(train_dataset, n_restarts = 1, tol=1e-2,
            tuning_method = "L-BFGS-B")
    else:
        model.tune_hyperparams_crude(train_dataset)

    print(f"Hyperparams, {model.kernel_choice}: {model.get_hyperparams()}")
    model.num_rffs = 8192

    model.fit(train_dataset, max_iter = 500, tol = 1e-6,  mode = "cg")
    cg_score = evaluate_model(model, train_dataset, test_dataset,
            False)

    print(f"CG score, {model.device}, {model.kernel_choice}: {cg_score}")

    model.num_rffs = 2048

    model.fit(train_dataset, mode = "exact")
    exact_score = evaluate_model(model, train_dataset, test_dataset,
            False)
    print(f"Exact score, {model.device}, {model.kernel_choice}: {exact_score}")

    return cg_score, exact_score



if __name__ == "__main__":
    unittest.main()
