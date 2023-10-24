"""Tests grid_bfgs fitting to ensure we achieve performance
>= what has been seen in the past for a similar # of RFFs and
kernel. Tests either CG or exact fitting."""
import sys

#TODO: Get rid of this path alteration
sys.path.append("..")
from utils.build_test_dataset import build_test_dataset
from utils.model_constructor import get_models
from utils.evaluate_model import evaluate_model


RANDOM_STATE = 123


def test_fit(kernel, conv_kernel, random_seed, conv_width = 3,
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
    if model.get_hyperparams().shape[0] < 4:
        model.tune_hyperparams_crude_bayes(train_dataset)
    elif model.get_hyperparams().shape[0] == 4:
        model.tune_hyperparams_crude_bayes(train_dataset, max_bayes_iter = 50)
    else:
        model.tune_hyperparams_crude_lbfgs(train_dataset, n_restarts = 1)

    print(f"Hyperparams, cpu, {kernel}: {model.get_hyperparams()}")

    hparams = model.get_hyperparams()
    model.num_rffs = cg_fitting_rffs
    model.initialize(train_dataset, RANDOM_STATE, hyperparams=hparams)

    if kernel == "Linear":
        preconditioner, _ = model.build_preconditioner(train_dataset,
            max_rank = 24, method = "srht")
    else:
        preconditioner, _ = model.build_preconditioner(train_dataset,
            max_rank = 256, method = "srht")

    model.fit(train_dataset,  preconditioner = preconditioner,
                max_iter = 500, random_seed = random_seed,
                tol = 1e-6,  mode = "cg")
    cg_score = evaluate_model(model, train_dataset, test_dataset,
            get_var)

    print(f"CG score, cpu, {kernel}: {cg_score}")

    model.num_rffs = exact_fitting_rffs
    model.initialize(train_dataset, RANDOM_STATE, hyperparams = hparams)

    model.fit(train_dataset,  random_seed = random_seed, mode = "exact")
    exact_score = evaluate_model(model, train_dataset, test_dataset,
            get_var)
    print(f"Exact score, cpu, {kernel}: {exact_score}")

    return cg_score, exact_score
