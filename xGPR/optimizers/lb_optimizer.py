"""Handles optimization for hyperparameters shared across
all kernels using a "telescoping grid" strategy used by
the minimal_bayes and grid optimization schemes,
scoring hyperparameters using NMLL."""
from copy import deepcopy
try:
    import cupy as cp
except:
    pass
import numpy as np



def shared_hparam_search(sigma_vals, kernel, dataset, init_bounds,
        n_pts_per_dim = 10, n_cycles = 3, subsample = 1,
        eigval_quotient = 1e6, min_eigval = 1e-6):
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
        random_seed (int): A seed for the random number generator.
    subsample (float): A value in the range [0.01,1] that indicates what
        fraction of the training set to use each time the gradient is
        calculated (the same subset is used every time). In general, 1
        will give better results, but using a subsampled subset can be
        a fast way to find the (approximate) location of a good
        hyperparameter set.
    eigval_quotient (float): A value by which the largest singular value
        of Z^T Z is divided to determine the smallest acceptable eigenvalue
        of Z^T Z (singular vectors with eigenvalues smaller than this
        are discarded). Setting this to larger values will make crude_bayes
        slightly more accurate, at the risk of numerical stability issues.
        In general, do not change this without good reason.
    min_eigval (float): If the largest singular value of Z^T Z divided by
        eigval_quotient is < min_eigval, min_eigval is used as the cutoff
        threshold instead. Setting this to smaller values will make crude_
        bayes slightly more accurate, at the risk of numerical stability
        issues. In general, do not change this without good reason.

    Returns:
        score (float): The NMLL associated with the best lambda-beta values found.
        best_lb (np.ndarray): The best lambda and beta values found on optimization.
    """
    bounds = init_bounds.copy()
    #VERY important: the calculations rely on the second shared hyperparameter
    #(hparams[1]) being set to np.log(1) = 0.
    #Do not change or remove the next four lines.
    hparams = np.zeros((sigma_vals.shape[0] + 2))
    if hparams.shape[0] > 2:
        hparams[2:] = sigma_vals
    kernel.set_hyperparams(hparams, logspace = True)

    eigvals, eigvecs, y_trans_y, ndatapoints = get_eigvals(kernel, dataset,
                        eigval_quotient = eigval_quotient,
                        min_eigval = min_eigval,
                        subsample = subsample)

    for i in range(n_cycles):
        lambda_, beta_, spacing = get_grid_pts(bounds, n_pts_per_dim, kernel.device)

        scoregrid = generate_scoregrid(dataset, kernel, eigvals, eigvecs,
                            lambda_, beta_, y_trans_y, ndatapoints)

        min_pt = scoregrid.argmin()
        best_score, best_lb = scoregrid[min_pt], [float(lambda_[min_pt]),
                                float(beta_[min_pt])]
        best_lb[0], best_lb[1] = np.log(best_lb[0]), np.log(best_lb[1])
        bounds[0,0] = max(best_lb[0] - spacing[0], init_bounds[0,0])
        bounds[0,1] = min(best_lb[0] + spacing[0], init_bounds[0,1])
        bounds[1,0] = max(best_lb[1] - spacing[1], init_bounds[1,0])
        bounds[1,1] = min(best_lb[1] + spacing[1], init_bounds[1,1])

    return np.round(float(best_score), 3), np.round(np.asarray(best_lb), 7)




def get_eigvals(kernel, dataset, eigval_quotient = 1e6,
        min_eigval = 1e-6, subsample = 1):
    """Gets the eigenvalues and eigenvectors of the z_trans_z
    matrix. Note that this matrix may be ill-conditioned.
    These are required for the NMLL procedure, not for
    multifitting."""
    if kernel.device == "gpu":
        z_trans_z = cp.zeros((kernel.get_num_rffs(), kernel.get_num_rffs()))
        z_trans_y = cp.zeros((kernel.get_num_rffs()))
    else:
        z_trans_z = np.zeros((kernel.get_num_rffs(), kernel.get_num_rffs()))
        z_trans_y = np.zeros((kernel.get_num_rffs()))

    y_trans_y = 0.0
    ndatapoints = 0

    if subsample == 1:
        for xdata, ydata in dataset.get_chunked_data():
            xtrans = kernel.transform_x(xdata)
            z_trans_z += xtrans.T @ xtrans
            z_trans_y += xtrans.T @ ydata
            y_trans_y += ydata.T @ ydata
            ndatapoints += xdata.shape[0]
    else:
        rng = np.random.default_rng(123)
        for xdata, ydata in dataset.get_chunked_data():
            idx_size = max(1, int(subsample * xdata.shape[0]))
            idx = rng.choice(xdata.shape[0], idx_size, replace=False)
            xdata, ydata = xdata[idx,...], ydata[idx]
            xdata = kernel.transform_x(xdata)
            z_trans_y += xdata.T @ ydata
            z_trans_z += xdata.T @ xdata
            y_trans_y += ydata.T @ ydata
            ndatapoints += xdata.shape[0]

    z_trans_z.flat[::z_trans_z.shape[0]+1] += 1e-5
    if kernel.device == "gpu":
        eigvecs, eigvals, _ = cp.linalg.svd(z_trans_z, full_matrices = False)
    else:
        eigvecs, eigvals, _ = np.linalg.svd(z_trans_z, full_matrices = False)

    #Get rid of the shift we added earlier for numerical stability.
    eigvals -= 1e-5

    #Small singular values are seldom reliable when the matrix is ill-conditioned;
    #eliminate these and associated singular vectors.
    adj_min_eigval = max(eigvals[0] / eigval_quotient, min_eigval)

    mask = eigvals > adj_min_eigval
    cut_point = max(mask.sum(), 1)
    eigvals[cut_point:] = 1e-14
    eigvecs[:,cut_point:] = 0
    eigvecs = eigvecs.T @ z_trans_y
    return eigvals, eigvecs, y_trans_y, ndatapoints




def generate_scoregrid(dataset, kernel, eigvals, eigvecs, lambda_,
                    beta_, y_trans_y, ndatapoints):
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
    if kernel.device == "cpu":
        logfun = np.log
    else:
        logfun = cp.log

    mask = eigvals > 0
    eigval_batch = eigvals[:,None] + lambda_[None,:]**2 / beta_[None,:]**2
    eigval_batch[eigval_batch < 1e-14] = 1e-14

    scoregrid = y_trans_y - eigvecs.T @ (mask[:,None] * \
            (eigvecs[:,None] / eigval_batch))

    #The best possible for y^T Z (lambda**2 + Z^T Z)^-1 Z^T y is when
    #Z (lambda**2 + Z^T Z)^-1 Z^T y = Zw = y, i.e. the fit is perfect,
    #in which case y^T y - y^T Z (lambda**2 + Z^T Z)^-1 Z^T y = 0.
    #It is possible to get negative values due to rounding error for
    #highly ill-conditioned matrices, hence, we clip at zero.
    scoregrid[scoregrid < 0] = 0

    scoregrid *= 0.5 / lambda_**2
    scoregrid += 0.5 * logfun(eigval_batch).sum(axis=0)
    scoregrid += kernel.get_num_rffs() * logfun(beta_)

    scoregrid += (ndatapoints - kernel.get_num_rffs()) * logfun(lambda_)
    scoregrid += ndatapoints * 0.5 * np.log(2 * np.pi)
    if kernel.device == "gpu":
        scoregrid = cp.asnumpy(scoregrid)
    return scoregrid


def get_grid_pts(bounds, n_pts_per_dim, device = "cpu"):
    """Generates grid points for the two shared hyperparameters
    for a given set of boundaries. Used for both multifitting and
    NMLL scoring.

    Args:
        bounds (np.ndarray): A 2 x 2 array where [0,:] is the boundaries
            for shared hyperparameter lambda and [1,:] is the boundaries
            for shared hyperparameter beta.
        n_pts_per_dim (int): The number of points per grid dimension.
        device (str): Either "cpu" or "gpu". Indicates where the values should
            be generated.

    Returns:
        lambda_ (np.ndarray): The grid coordinates for the first hyperparameter.
        beta_ (np.ndarray): The grid coordinates for the second hyperparameter.
        spacing (array): The spacing between grid points.
    """
    lambda_pts = np.linspace(bounds[0,0], bounds[0,1], n_pts_per_dim)
    beta_pts = np.linspace(bounds[1,0], bounds[1,1], n_pts_per_dim)
    lambda_pts, beta_pts = np.meshgrid(lambda_pts, beta_pts)
    lambda_ = np.exp(lambda_pts.flatten())
    beta_ = np.exp(beta_pts.flatten())

    spacing = np.zeros((2))
    spacing[0] = 1.05 * np.abs(bounds[0,0] - bounds[0,1]) / n_pts_per_dim
    spacing[1] = 1.05 * np.abs(bounds[1,0] - bounds[1,1]) / n_pts_per_dim
    if device == "gpu":
        lambda_, beta_ = cp.asarray(lambda_), cp.asarray(beta_)
    return lambda_, beta_, spacing
