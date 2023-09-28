"""Tests grid_bfgs fitting to ensure we achieve performance
>= what has been seen in the past for a similar # of RFFs and
kernel. Tests either CG or exact fitting."""
import sys

#TODO: Get rid of this path alteration
sys.path.append("..")
from utils.build_test_dataset import build_test_dataset
from utils.model_constructor import get_models
from utils.evaluate_model import evaluate_model



def test_fit_cpu(kernel, conv_kernel, random_seed, conv_width = 3,
            get_var = True, conv_ard_kernel = False,
            training_rffs = 512, cg_fitting_rffs = 8192,
            exact_fitting_rffs = 2048):
    """Test on CPU using preconditioned CG and exact."""
    _, train_dataset = build_test_dataset(conv_kernel = conv_kernel)
    cpu_mod, _ = get_models(kernel, train_dataset.get_xdim(), conv_width,
                            conv_ard_kernel = conv_ard_kernel,
                            training_rffs = training_rffs,
                            fitting_rffs = exact_fitting_rffs)
    cpu_mod.training_rffs = training_rffs
    cpu_mod.fitting_rffs = cg_fitting_rffs
    cpu_mod.verbose = False

    test_dataset, _ = build_test_dataset(conv_kernel = conv_kernel,
            xsuffix = "testxvalues.npy", ysuffix = "testyvalues.npy")
    if cpu_mod.get_hyperparams().shape[0] < 4:
        cpu_mod.tune_hyperparams_crude_bayes(train_dataset)
    elif cpu_mod.get_hyperparams().shape[0] == 4:
        cpu_mod.tune_hyperparams_crude_bayes(train_dataset, max_bayes_iter = 50)
    else:
        cpu_mod.tune_hyperparams_crude_lbfgs(train_dataset, n_restarts = 1)

    print(f"Hyperparams, cpu, {kernel}: {cpu_mod.get_hyperparams()}")

    if kernel == "Linear":
        preconditioner, _ = cpu_mod.build_preconditioner(train_dataset,
            max_rank = 24, method = "srht")
    else:
        preconditioner, _ = cpu_mod.build_preconditioner(train_dataset,
            max_rank = 256, method = "srht")

    cpu_mod.fit(train_dataset,  preconditioner = preconditioner,
                max_iter = 500, random_seed = random_seed,
                tol = 1e-6,  mode = "cg")
    cg_score = evaluate_model(cpu_mod, train_dataset, test_dataset,
            get_var)

    print(f"CG score, cpu, {kernel}: {cg_score}")

    cpu_mod.fitting_rffs = exact_fitting_rffs
    cpu_mod.fit(train_dataset,  random_seed = random_seed, mode = "exact")
    exact_score = evaluate_model(cpu_mod, train_dataset, test_dataset,
            get_var)
    print(f"Exact score, cpu, {kernel}: {exact_score}")

    return cg_score, exact_score



def test_fit_gpu(kernel, conv_kernel, random_seed, conv_width = 3,
            get_var = True, conv_ard_kernel = False,
            training_rffs = 512, cg_fitting_rffs = 8192,
            exact_fitting_rffs = 2048):
    """Test on GPU using preconditioned CG and exact fitting."""
    _, train_dataset = build_test_dataset(conv_kernel = conv_kernel)
    _, gpu_mod = get_models(kernel, train_dataset.get_xdim(), conv_width,
                            conv_ard_kernel = conv_ard_kernel,
                            training_rffs = training_rffs,
                            fitting_rffs = exact_fitting_rffs)
    if gpu_mod is None:
        #If GPU not available, return immediately.
        return None, None

    gpu_mod.training_rffs = training_rffs
    gpu_mod.fitting_rffs = cg_fitting_rffs
    gpu_mod.verbose = False

    test_dataset, _ = build_test_dataset(conv_kernel = conv_kernel,
            xsuffix = "testxvalues.npy", ysuffix = "testyvalues.npy")

    if gpu_mod.get_hyperparams().shape[0] < 4:
        gpu_mod.tune_hyperparams_crude_bayes(train_dataset)
    elif gpu_mod.get_hyperparams().shape[0] == 4:
        gpu_mod.tune_hyperparams_crude_bayes(train_dataset, max_bayes_iter = 50)
    else:
        gpu_mod.tune_hyperparams_crude_lbfgs(train_dataset, n_restarts = 1)

    print(f"Hyperparams, gpu, {kernel}: {gpu_mod.get_hyperparams()}")

    if kernel == "Linear":
        preconditioner, _ = gpu_mod.build_preconditioner(train_dataset,
                max_rank = 24, method = "srht")
    else:
        preconditioner, _ = gpu_mod.build_preconditioner(train_dataset,
                max_rank = 256, method = "srht")

    gpu_mod.fit(train_dataset,  preconditioner = preconditioner,
                max_iter = 500, random_seed = random_seed,
                tol = 1e-6,  mode = "cg")
    cg_score = evaluate_model(gpu_mod, train_dataset, test_dataset,
                get_var)

    print(f"CG score, gpu, {kernel}: {cg_score}")
    ######################
    import pickle
    out_data = {"model":gpu_mod, "kernel":kernel, "dset":train_dataset}
    with open(f"{kernel}.pk", "wb") as fhandle:
        pickle.dump(out_data, fhandle)
    #########################

    gpu_mod.fitting_rffs = exact_fitting_rffs
    gpu_mod.fit(train_dataset,  random_seed = random_seed, mode = "exact")
    exact_score = evaluate_model(gpu_mod, train_dataset, test_dataset,
                get_var)


    print(f"Exact score, gpu, {kernel}: {exact_score}")
    return cg_score, exact_score
