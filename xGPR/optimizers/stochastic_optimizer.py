"""Contains the tools needed to perform amsgrad SGD using
MAP training set loss as the objective (rather than NMLL),
an experimental approach."""
try:
    import cupy as cp
except:
    pass
import numpy as np

from ..scoring_tools.gradient_tools import minibatch_map_gradient


def amsgrad_optimizer(kernel, init_param_vals, dataset, bounds,
                        minibatch_size = 1000,
                        n_epochs = 2, learn_rate = 0.001,
                        a_reg = 1.0, verbose = True):
    """Performs a single round of AMSGrad stochastic gradient descent optimization
    for the NMLL (for hyperparameter tuning).

    Args:
        kernel: A valid kernel object that can generate random features.
        init_param_vals (np.ndarray): A starting point for this round of
            optimization.
        dataset: An OnlineDataset or OfflineDataset with the raw data we will
            use for tuning.
        bounds (np.ndarray): The boundaries of the hyperparameter optimization.
            Must be of shape N x 2 for N hyperparameters.
        minibatch_size (int): The minibatch size.
        n_epochs (int): The number of epochs per restart.
        learn_rate (float): The learning rate parameter.
        a_reg (float): A regularization hyperparameter that must be > 0.
        verbose (bool): If True, regular updates are printed during optimization.

    Returns:
        params (np.ndarray): The best hyperparameters obtained.
        iteration (int): The number of iterations performed.
    """
    average_now = False
    optim_eps = 1e-8
    beta1, beta2 = 0.9, 0.999
    params = np.zeros((init_param_vals.shape[0] + kernel.get_num_rffs()))
    params[:init_param_vals.shape[0]] = init_param_vals

    m_t, v_t = np.zeros((params.shape[0])), \
                np.zeros((2, params.shape[0]))
    if kernel.device == "gpu":
        m_t, v_t, params = cp.asarray(m_t), cp.asarray(v_t), cp.asarray(params)

    iteration = 0
    for epoch in range(n_epochs):
        if epoch >= n_epochs - 1:
            average_now, n_average_iterations = True, 1
            best_params = params.copy()

        end_epoch = False
        dataset.reset_index()

        while not end_epoch:
            xbatch, ybatch, end_epoch = dataset.get_next_minibatch(minibatch_size)

            grad = minibatch_map_gradient(params, xbatch, ybatch, kernel, a_reg)
            m_t[:] = beta1 * m_t + (1 - beta1) * grad
            v_t[0,:] = beta2 * v_t[1,:] + (1 - beta2) * grad**2
            v_t[0,:] = np.max(v_t, axis=0)
            v_t[1,:] = v_t[0,:]

            params -= learn_rate * m_t / (np.sqrt(v_t[0,:] + optim_eps))
            params[:bounds.shape[0]] = params[:bounds.shape[0]].clip(min=bounds[:,0],
                                max=bounds[:,1])

            if average_now:
                best_params *= (n_average_iterations - 1) / n_average_iterations
                best_params += params / n_average_iterations
                n_average_iterations += 1
            iteration += 1
            if iteration % 20 == 0 and verbose:
                print(f"Iteration {iteration} complete")
                print(f"Grad norm**2: {(grad * grad).sum()}")
        if verbose:
            print(f"Epoch {epoch + 1} ended")

    if kernel.device == "gpu":
        params = cp.asnumpy(params)

    if not average_now:
        return params, epoch

    return best_params, epoch
