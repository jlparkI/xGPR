"""Handles optimization for hyperparameters shared across
all kernels using a "telescoping grid" strategy used by
the minimal_bayes and grid optimization schemes,
scoring hyperparameters using NMLL."""
try:
    import cupy as cp
except:
    pass
import numpy as np



def shared_hparam_search(sigma_vals, kernel, dataset, init_bounds,
        n_pts_per_dim = 100, n_cycles = 1, subsample = 1):
    """Uses a "telescoping grid" procedure to search the space of
    shared hyperparameters for a given set of kernel specific hyperparameters
    (sigma_vals), using exact NMLL to score.

    Args:
        sigma_vals (np.ndarray): A set of kernel-specific hyperparameters,
            of shape (# hyperparameters - 2). Should not be shape > 2.
        kernel: A valid kernel object that can generate random features.
        dataset: A valid dataset object that can retrieve chunked data.
        init_bounds (np.ndarray): A 2 x 2 array where [0,:] is the boundaries
            for shared hyperparameter lambda and [1,:] is the boundaries
            for shared hyperparameter beta.
        n_pts_per_dim (int): The number of points per grid dimension.
        n_cycles (int): The number of 'telescoping grid' cycles.
        subsample (float): A value in the range [0.01,1] that indicates what
            fraction of the training set to use each time the gradient is
            calculated (the same subset is used every time). In general, 1
            will give better results, but using a subsampled subset can be
            a fast way to find the (approximate) location of a good
            hyperparameter set.

    Returns:
        score (float): The NMLL associated with the best lambda-beta values found.
        best_lb (np.ndarray): The best lambda and beta values found on optimization.
    """
    bounds = init_bounds.copy()
    if np.exp(bounds[0,0]) < 1e-3:
        bounds[0,0] = np.log(1e-3)

    hparams = np.zeros((sigma_vals.shape[0] + 1))
    if hparams.shape[0] > 1:
        hparams[1:] = sigma_vals
    kernel.set_hyperparams(hparams, logspace = True)

    eigvals, eigvecs, y_trans_y, ndatapoints = get_eigvals(kernel, dataset,
                        subsample = subsample)

    for _ in range(n_cycles):
        lambda_, spacing = get_grid_pts(bounds, n_pts_per_dim, kernel.device)

        scoregrid = generate_scoregrid(kernel, eigvals, eigvecs,
                            lambda_, y_trans_y, ndatapoints)

        min_pt = scoregrid.argmin()
        best_score, best_lb = scoregrid[min_pt], np.log(float(lambda_[min_pt]))
        bounds[0,0] = max(best_lb - spacing, init_bounds[0,0])
        bounds[0,1] = min(best_lb + spacing, init_bounds[0,1])

    return np.round(float(best_score), 3), np.round(np.asarray([best_lb]), 7)




def get_eigvals(kernel, dataset, subsample = 1):
    """Gets the eigenvalues and eigenvectors of the z_trans_z
    matrix. Note that this matrix may be ill-conditioned.
    These are required for the NMLL procedure, not for
    multifitting."""
    if kernel.device == "cuda":
        z_trans_z = cp.zeros((kernel.get_num_rffs(), kernel.get_num_rffs()))
        z_trans_y = cp.zeros((kernel.get_num_rffs()))
    else:
        z_trans_z = np.zeros((kernel.get_num_rffs(), kernel.get_num_rffs()))
        z_trans_y = np.zeros((kernel.get_num_rffs()))

    y_trans_y = 0.0
    ndatapoints = 0

    if subsample == 1:
        for xin, yin, ldata in dataset.get_chunked_data():
            xtrans, ydata = kernel.transform_x_y(xin, yin, ldata)
            z_trans_z += xtrans.T @ xtrans
            z_trans_y += xtrans.T @ ydata
            y_trans_y += ydata.T @ ydata
            ndatapoints += xin.shape[0]
    else:
        rng = np.random.default_rng(123)
        for xin, yin, ldata in dataset.get_chunked_data():
            idx_size = max(1, int(subsample * xin.shape[0]))
            idx = rng.choice(xin.shape[0], idx_size, replace=False)
            xin, yin, ldata = xin[idx,...], yin[idx], ldata[idx]
            xdata, ydata = kernel.transform_x(xin, yin, ldata)

            z_trans_y += xdata.T @ ydata
            z_trans_z += xdata.T @ xdata
            y_trans_y += ydata.T @ ydata
            ndatapoints += xdata.shape[0]

    z_trans_z.flat[::z_trans_z.shape[0]+1] += 1e-5
    if kernel.device == "cuda":
        eigvecs, eigvals, _ = cp.linalg.svd(z_trans_z, full_matrices = False)
    else:
        eigvecs, eigvals, _ = np.linalg.svd(z_trans_z, full_matrices = False)

    #Subtract out the additional 1e-5 for numerical stability.
    eigvals -= 1e-5

    mask = eigvals >= 1e-7
    cut_point = max(mask.sum(), 1)
    eigvals[cut_point:] = 1e-7
    eigvecs[:,cut_point:] = 0
    eigvecs = eigvecs.T @ z_trans_y
    return eigvals, eigvecs, y_trans_y, ndatapoints




def generate_scoregrid(kernel, eigvals, eigvecs, lambda_,
                    y_trans_y, ndatapoints):
    """Generates scores for all the lambda / beta values that
    are part of the current grid, using NMLL exact scoring.

    Args:
        dataset: A valid dataset object that can generate chunked data.
        kernel: A valid kernel object that can generate random features.
        eigvals (array): A (x) array of eigenvalues.
        eigvecs (array): A (x) array of eigenvectors.T @ (z^T @ y).
        lambda_ (array): A (num_params) array of lambda values.
        beta_ (array): A (num_params) array of beta values.
        y_trans_y (float): The quantity y^T y.

    Returns:
        scoregrid (array): A (num_params) array of scores.
    """
    eigval_batch = eigvals[:,None] + lambda_[None,:]**2
    scoregrid = y_trans_y - eigvecs.T @ (eigvecs[:,None] / eigval_batch)

    #The best possible for y^T Z (lambda**2 + Z^T Z)^-1 Z^T y is when
    #Z (lambda**2 + Z^T Z)^-1 Z^T y = Zw = y, i.e. the fit is perfect,
    #in which case y^T y - y^T Z (lambda**2 + Z^T Z)^-1 Z^T y = 0.
    #It is possible to get negative values due to rounding error for
    #highly ill-conditioned matrices, hence, we clip at zero.
    scoregrid[scoregrid < 0] = 0

    scoregrid *= 0.5
    if kernel.device == "cpu":
        logfun = np.log
        beta = np.sqrt(2 * scoregrid / (ndatapoints * lambda_**2))
    else:
        logfun = cp.log
        beta = cp.sqrt(2 * scoregrid / (ndatapoints * lambda_**2))

    #Allowing beta to take ANY value does not affect numerical stability but misrepresents
    #the outcome that would occur if a particular lambda_ value were selected.
    beta = beta.clip(min=0.1, max=10)

    scoregrid  /= (beta * lambda_)**2
    scoregrid += 0.5 * logfun(eigval_batch).sum(axis=0)

    scoregrid += (ndatapoints - kernel.get_num_rffs()) * logfun(lambda_)
    scoregrid += ndatapoints * 0.5 * np.log(2 * np.pi) + ndatapoints * logfun(beta)
    if kernel.device == "cuda":
        scoregrid = cp.asnumpy(scoregrid)
    return scoregrid


def get_grid_pts(bounds, n_pts_per_dim, device = "cpu"):
    """Generates grid points for the shared hyperparameters
    for a given set of boundaries. Used for both multifitting and
    NMLL scoring.

    Args:
        bounds (np.ndarray): A 1 x 2 array where [0,:] is the boundaries
            for shared hyperparameter lambda.
        n_pts_per_dim (int): The number of points per grid dimension.
        device (str): Either "cpu" or "cuda". Indicates where the values should
            be generated.

    Returns:
        lambda_ (np.ndarray): The grid coordinates for the first hyperparameter.
        spacing (array): The spacing between grid points.
    """
    lambda_pts = np.exp(np.linspace(bounds[0,0], bounds[0,1], n_pts_per_dim))

    spacing = 1.05 * np.abs(bounds[0,0] - bounds[0,1]) / n_pts_per_dim
    if device == "cuda":
        lambda_pts = cp.asarray(lambda_pts)
    return lambda_pts, spacing
