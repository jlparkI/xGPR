"""Tests grid_bfgs fitting to ensure we achieve performance
>= what has been seen in the past for a similar # of RFFs and
kernel. Tests either CG or exact fitting."""
import sys

#TODO: Get rid of this path alteration
sys.path.append("..")
from utils.build_test_dataset import build_test_dataset
from utils.model_constructor import get_models
from utils.evaluate_model import evaluate_model

def test_fit_cg(kernel, conv_kernel, random_seed, conv_width = 3,
            get_var = True):
    """Test using preconditioned cg."""
    _, train_dataset = build_test_dataset(conv_kernel = conv_kernel)
    cpu_mod, gpu_mod = get_models(kernel, train_dataset.get_xdim(), conv_width)
    cpu_mod.training_rffs = 512
    cpu_mod.fitting_rffs = 16384
    cpu_mod.verbose = False

    test_dataset, _ = build_test_dataset(conv_kernel = conv_kernel,
            xsuffix = "testxvalues.npy", ysuffix = "testyvalues.npy")
    cpu_mod.tune_hyperparams_crude_bayes(train_dataset)
    print(f"Hyperparams, cpu, {kernel}: {cpu_mod.get_hyperparams()}")

    preconditioner, _ = cpu_mod.build_preconditioner(train_dataset,
            max_rank = 256, method = "srht")

    cpu_mod.fit(train_dataset,  preconditioner = preconditioner,
                max_iter = 500, random_seed = random_seed,
                tol = 1e-6,  mode = "cg")
    cpu_score = evaluate_model(cpu_mod, train_dataset, test_dataset,
            get_var)

    print(f"CG score, cpu, {kernel}: {cpu_score}")

    if gpu_mod is not None:
        gpu_mod.training_rffs = 512
        gpu_mod.fitting_rffs = 16384
        gpu_mod.verbose = False

        gpu_mod.tune_hyperparams_crude_bayes(train_dataset)
        print(f"Hyperparams, gpu, {kernel}: {gpu_mod.get_hyperparams()}")
        preconditioner, _ = gpu_mod.build_preconditioner(train_dataset,
                max_rank = 256, method = "srht")

        gpu_mod.fit(train_dataset,  preconditioner = preconditioner,
                max_iter = 500, random_seed = random_seed,
                tol = 1e-6,  mode = "cg")
        gpu_score = evaluate_model(gpu_mod, train_dataset, test_dataset,
                get_var)

        print(f"CG score, gpu, {kernel}: {gpu_score}")
        return cpu_score, gpu_score
    return cpu_score, None


def test_fit_exact(kernel, conv_kernel, random_seed, conv_width = 3,
        get_var = True):
    """Test using preconditioned cg."""
    _, train_dataset = build_test_dataset(conv_kernel = conv_kernel)
    cpu_mod, gpu_mod = get_models(kernel, train_dataset.get_xdim(), conv_width)
    cpu_mod.training_rffs = 512
    cpu_mod.fitting_rffs = 2048
    cpu_mod.verbose = False

    test_dataset, _ = build_test_dataset(conv_kernel = conv_kernel,
            xsuffix = "testxvalues.npy", ysuffix = "testyvalues.npy")

    cpu_mod.tune_hyperparams_crude_bayes(train_dataset)
    print(f"Hyperparams, cpu, {kernel}: {cpu_mod.get_hyperparams()}")
    cpu_mod.fit(train_dataset,  random_seed = random_seed, mode = "exact")
    cpu_score = evaluate_model(cpu_mod, train_dataset, test_dataset,
            get_var)

    print(f"Exact score, cpu, {kernel}: {cpu_score}")

    if gpu_mod is not None:
        gpu_mod.training_rffs = 512
        gpu_mod.fitting_rffs = 2048
        gpu_mod.verbose = False

        gpu_mod.tune_hyperparams_crude_bayes(train_dataset)
        print(f"Hyperparams, gpu, {kernel}: {gpu_mod.get_hyperparams()}")
        gpu_mod.fit(train_dataset,  random_seed = random_seed, mode = "exact")
        gpu_score = evaluate_model(gpu_mod, train_dataset, test_dataset,
                get_var)

        print(f"Exact score, gpu, {kernel}: {gpu_score}")
        return cpu_score, gpu_score
    return cpu_score, None
