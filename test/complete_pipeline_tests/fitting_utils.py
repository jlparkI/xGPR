"""Tests grid_bfgs fitting to ensure we achieve performance
>= what has been seen in the past for a similar # of RFFs and
kernel. Tests either CG or exact fitting."""
import sys
import optuna
import numpy as np

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
    nhparams = model.get_hyperparams().shape[0]
    if nhparams < 3:
        #_, _, score = model.tune_hyperparams_lbfgs(train_dataset, n_restarts = 1)
        _, _, score = model.tune_hyperparams_crude(train_dataset)
        print(score)
    else:
        model.tune_hyperparams_lbfgs(train_dataset, n_restarts = 1)

    hparams = model.get_hyperparams()

    print(f"Hyperparams, cpu, {kernel}: {model.get_hyperparams()}")

    model.num_rffs = cg_fitting_rffs
    model.set_hyperparams(hparams, train_dataset)

    if kernel == "Linear":
        preconditioner, _ = model.build_preconditioner(train_dataset,
            max_rank = 24, method = "srht")
    else:
        preconditioner, _ = model.build_preconditioner(train_dataset,
            max_rank = 256, method = "srht")

    model.fit(train_dataset,  preconditioner = preconditioner,
                max_iter = 500, tol = 1e-6,  mode = "cg")
    cg_score = evaluate_model(model, train_dataset, test_dataset,
            get_var)

    print(f"CG score, cpu, {kernel}: {cg_score}")

    model.num_rffs = exact_fitting_rffs
    model.set_hyperparams(hparams, train_dataset)

    model.fit(train_dataset, mode = "exact")
    exact_score = evaluate_model(model, train_dataset, test_dataset,
            get_var)
    print(f"Exact score, cpu, {kernel}: {exact_score}")

    return cg_score, exact_score


def single_hparam_objective(trial, model, dataset):
    """Wrapper on NMLL calculations enabling Optuna to tune."""
    params = {"lambda":trial.suggest_float('lambda', 1e-6, 10, log=True)}
    hparams = np.log(np.array([params['lambda']]))
    return model.exact_nmll(hparams, dataset)

def two_hparam_objective(trial, model, dataset):
    """Wrapper on NMLL calculations enabling Optuna to tune."""
    params = {"lambda":trial.suggest_float('lambda', 1e-6, 10, log=True),
            "sigma":trial.suggest_float('sigma', 1e-5, 10, log=True)}
    hparams = np.log(np.array([params['lambda'], params['sigma']]))
    return model.exact_nmll(hparams, dataset)
