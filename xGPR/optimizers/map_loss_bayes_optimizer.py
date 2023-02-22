"""Describes the map_loss bayesian tuning routine, which uses Bayesian
optimization to optimize a hyperprior."""
import warnings
import copy

import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF

from .stochastic_optimizer import amsgrad_optimizer
from ..scoring_toolkit.map_gradient_tools import full_map_gradient


def bayes_map_loss_tuning(kernel, dataset, areg_bounds, optimizer_args,
                    approx_cost_fun, random_seed, max_iter, tol = 1e-1,
                    n_init_pts = 10):
    """Conducts Bayesian optimization using MAP loss on the training set,
    followed by approximate NMLL calculation.

    Args:
        kernel: A valid kernel that can generate random features.
        dataset: An OnlineDataset or OfflineDataset containing raw data.
        bounds (np.ndarray): An N x 2 for N hyperparameters set of boundaries
            for optimization.
        a_reg_bounds (np.ndarray): Boundaries on the hyperprior; shape should
            be (1 x 2).
        optimizer_args (dict): A dictionary of arguments to the optimizer. Should
            contain:
            'bounds' (an N x 2 array for N hyperparameters bounding the
            hyperparameter search)
            'minibatch_size' (an int determining minibatch size)
            'n_restarts' (an int determining number of restarts),
            'max_epochs' (the max number of epochs)
            'mode' (one of 'sgd', 'lbfgs' determining optimizer type)
            'learn_rate' (the learning rate for SGD).
            'nmll_rank', 'nmll_probes', 'nmll_iter', 'nmll_tol': Additional arguments
                that control calculation of the approximate NMLL.
            Some of these arguments are ignored by the 'lbfgs' optimizer.
        approx_cost_fun: A function this routine can call that evaluates the approximate NMLL.
        random_seed (int): A seed for the random number generator.
        max_iter (int): The maximum number of iterations.
        tol (float): The threshold for convergence.
        n_init_pts (int): The number of initial grid points to evaluate before
            Bayesian optimization. 10 (the default) is usually fine. If you are
            searcing a smaller space, however, you can sometimes save time by using
            a smaller # (e.g. 5).

    Returns:
        best_hparams (np.ndarray): The best hyperparameters obtained.
        res (tuple): A tuple of (areg_pts, scores), indicating the areg regularization
            values tested and the score obtained for each.
        best_score (float): The best NMLL obtained.
        total_iter (int): The total number of epochs executed.

    Raises:
        ValueError: A ValueError is raised if problematic values are supplied.
    """
    num_hparams = kernel.get_hyperparams().shape[0]
    areg_pts = np.linspace(areg_bounds[0, 0], areg_bounds[0, 1],
                                n_init_pts)
    areg_pts = list(np.round(areg_pts, 7))
    total_iter = 0

    hparams, scores = [], []

    for i, areg_pt in enumerate(areg_pts):
        score, hparam, niter = evaluate_areg_value(kernel, dataset, areg_pt,
                            approx_cost_fun, random_seed + i,
                            optimizer_args)
        total_iter += niter
        scores.append(score)
        hparams.append(hparam[:num_hparams])
        if optimizer_args["verbose"]:
            print(f"Grid point {i} acquired.")

    surrogate = GPR(kernel = RBF(),
            normalize_y = True,
            alpha = 1e-6, random_state = random_seed,
            n_restarts_optimizer = 4)


    for iternum in range(n_init_pts, max_iter):
        new_areg, min_dist, surrogate = propose_new_point(
                            areg_pts, scores, surrogate,
                            areg_bounds, random_seed + iternum)
        if optimizer_args["verbose"]:
            print(f"New areg: {new_areg}")
        score, hparam, niter = evaluate_areg_value(kernel, dataset, float(new_areg),
                            approx_cost_fun, random_seed + n_init_pts + iternum,
                            optimizer_args)
        total_iter += niter
        scores.append(score)
        areg_pts.append(float(new_areg))
        hparams.append(hparam[:num_hparams])

        if min_dist < tol:
            break
        if optimizer_args["verbose"]:
            print(f"Additional acquisition {iternum}.")

    best_hparams = hparams[np.argmin(scores)]
    if optimizer_args["verbose"]:
        print(f"Best score achieved: {np.round(np.min(scores), 4)}")
        print(f"Best hyperparams: {best_hparams}")
    return best_hparams, (areg_pts, scores), np.min(scores), total_iter


def propose_new_point(areg_vals, scores,
                surrogate, bounds, random_seed,
                num_cand = 500):
    """Refits the 'surrogate' model and uses it to propose new
    locations for exploration (via Thompson sampling).

    Args:
        sigma_vals (np.ndarray): A grid of kernel-specific hyperparameters
            at which NMLL has been evaluated so far.
        scores (array-like): The scores for sigma_vals.
        surrogate: A scikit-learn Gaussian process model with a Matern kernel.
            Is fitted to sigma_vals, scores.
        bounds (np.ndarray): An N x 2 for N kernel specific hyperparameters
            array of boundaries for optimization.
        random_seed (int): A seed for the random number generator.
        num_cand (int): The number of candidate points to evaluate.

    Returns:
        best_candidate (np.ndarray): A set of kernel-specific hyperparameters
            at which to propose the next acquisition.
        min_dist (float): The smallest distance between best_candidate
            and any existing candidates.
        surrogate: The Gaussian process acquisition model now refitted
            to the updated data.
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xvals = np.asarray(areg_vals).reshape(-1,1)
        surrogate.fit(xvals, scores)

    rng = np.random.default_rng(random_seed)
    candidates = rng.uniform(low=bounds[:,0], high=bounds[:,1],
                           size = (num_cand, bounds.shape[0]))
    candidates = np.round(candidates, 7)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_candidates = surrogate.sample_y(candidates, n_samples=15,
                    random_state=random_seed)
    best_idx = np.unravel_index(y_candidates.argmin(),
                            y_candidates.shape)
    best_cand = candidates[best_idx[0],:]
    min_dist = np.min(np.linalg.norm(best_cand[None,:] - xvals, axis=1))
    return candidates[best_idx[0],:], min_dist, surrogate




def evaluate_areg_value(kernel, dataset, a_reg, approx_cost_fun,
        random_seed, optim_args):
    """Fits the model for a specified value of a_reg and returns the best
    score and parameters achieved.

    Args:
        kernel: A valid kernel that can generate random features.
        dataset: An OnlineDataset or OfflineDataset containing raw data.
        a_reg_bounds (float): A regularization value. Smaller = tighter regularization.
        approx_cost_fun: A function this routine can call that evaluates the approximate NMLL.
        random_seed (int): A seed for the random number generator.
        optimizer_args (dict): A dictionary of arguments to the optimizer. Should
            contain:
            'bounds' (an N x 2 array for N hyperparameters bounding the
            hyperparameter search)
            'minibatch_size' (an int determining minibatch size)
            'n_restarts' (an int determining number of restarts),
            'max_epochs' (the max number of epochs)
            'mode' (one of 'sgd', 'lbfgs' determining optimizer type)
            'learn_rate' (the learning rate for SGD).
            'nmll_rank', 'nmll_probes', 'nmll_iter', 'nmll_tol': Additional arguments
                that control calculation of the approximate NMLL.
            Some of these arguments are ignored by the 'lbfgs' optimizer.

    Returns:
        best_score (float): The best NMLL achieved.
        best_x (np.ndarray): The best hyperparameters found.
        niter (int): The number of epochs required to achieve this.

    Raises:
        ValueError: A ValueError is raised if no solution is found.
    """
    rng = np.random.default_rng(random_seed)
    niter, best_score = 0, np.inf
    best_x = None

    a_reg_value = np.exp(a_reg)

    num_hparams = kernel.get_hyperparams().shape[0]
    init_params = np.empty((num_hparams + kernel.get_num_rffs()))

    bounds = optim_args["bounds"]
    if optim_args["mode"] == "lbfgs":
        args = (dataset, kernel, a_reg_value, optim_args["verbose"])
        bounds_tuples = list(map(tuple, bounds)) + [(None, None) for i in
                        range(kernel.get_num_rffs())]

    for i in range(optim_args["n_restarts"]):
        init_params[:] = 0
        init_hyperparams = np.array([rng.uniform(low = bounds[j,0], high = bounds[j,1])
                for j in range(num_hparams)])
        init_params[:init_hyperparams.shape[0]] = init_hyperparams

        if optim_args["mode"] == "lbfgs":
            res = minimize(full_map_gradient, options={"maxiter":optim_args["max_epochs"]},
                        x0 = init_params, args = args,
                        jac = True, bounds = bounds_tuples)
            params = res.x
            niter += res.nfev

        else:
            params = amsgrad_optimizer(kernel, init_params, dataset, bounds,
                        optim_args["minibatch_size"], optim_args["max_epochs"],
                        learn_rate = optim_args["learn_rate"],
                        a_reg = a_reg_value, verbose = optim_args["verbose"])
            niter += optim_args["max_epochs"]

        hparams = params[:num_hparams]
        cost = approx_cost_fun(hparams, dataset, optim_args["nmll_rank"],
                        optim_args["nmll_probes"], random_seed, optim_args["nmll_iter"],
                        optim_args["nmll_tol"])

        if cost < best_score:
            best_x = hparams.copy()
            best_score = copy.deepcopy(cost)
            if optim_args["verbose"]:
                print(f"New best hparams: {hparams}")

    if best_x is None:
        raise ValueError("All restarts failed to find acceptable hyperparameters.")
    return best_score, best_x, niter
