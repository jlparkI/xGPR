"""Contains the tools needed to perform adam stochastic gradient descent.
Note that in general this method achieves inferior results and we do
not recommend it except as a necessary evil for very large datasets
(or without some substantial modifications)."""
import copy
import numpy as np



def amsgrad_optimizer(grad_fun, init_param_vals, dataset, bounds,
                        minibatch_size = 2000,
                        n_epochs = 2, learn_rate = 0.001,
                        start_averaging = 1000, verbose = True):
    """Performs a single round of AMSGrad stochastic gradient descent optimization
    for the NMLL (for hyperparameter tuning).

    Args:
        cost_fun: A function that evaluates the negative marginal log likelihood
            AND its gradient for a given set of input hyperparameters and an
            input MinibatchDataset object.
        init_param_vals (np.ndarray): A starting point for this round of
            optimization.
        dataset: An OnlineDataset or OfflineDataset with the raw data we will
            use for tuning.
        bounds (np.ndarray): The boundaries of the hyperparameter optimization.
            Must be of shape N x 2 for N hyperparameters.
        minibatch_size (int): The minibatch size.
        n_epochs (int): The number of epochs per restart.
        learn_rate (float): The learning rate parameter for the Adam algorithm.
        start_averaging (int): Start averaging after this many epochs. If >
            n_epochs, never start averaging.
        verbose (bool): If True, regular updates are printed during optimization.

    Returns:
        params (np.ndarray): The best hyperparameters obtained.
        iteration (int): The number of iterations performed.
    """
    average_now = False
    optim_eps = 1e-8
    beta1, beta2 = 0.9, 0.999
    params = init_param_vals.copy()

    m_t, v_t = np.zeros((params.shape[0])), \
                np.zeros((2, params.shape[0]))
    iteration = 0
    for epoch in range(n_epochs):
        if epoch == start_averaging:
            average_now, n_average_iterations = True, 1
            best_params = params.copy()

        end_epoch = False
        dataset.reset_index()

        while not end_epoch:
            xbatch, ybatch, end_epoch = dataset.get_next_minibatch(minibatch_size)

            grad = grad_fun(params, xbatch, ybatch)
            m_t[:] = beta1 * m_t + (1 - beta1) * grad
            v_t[0,:] = beta2 * v_t[1,:] + (1 - beta2) * grad**2
            v_t[0,:] = np.max(v_t, axis=0)
            v_t[1,:] = v_t[0,:]

            params -= learn_rate * m_t / (np.sqrt(v_t[0,:] + optim_eps))
            params[:bounds.shape[0]] = np.clip(params[:bounds.shape[0]],
                        a_min=bounds[:,0], a_max=bounds[:,1])

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

    if not average_now:
        return params, epoch

    return best_params, epoch
