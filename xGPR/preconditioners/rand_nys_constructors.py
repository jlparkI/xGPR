"""Contains the tools necessary to generate the matrices required for
the preconditioner on either CPU or GPU, using several different
methods."""
try:
    import cupy as cp
    import cupyx
    from cupyx import scipy
    from cupyx.scipy import linalg
except:
    pass
import numpy as np
import scipy

from ..kernels.srht_compressor import SRHTCompressor



def single_pass_gauss(dataset, kernel, q_mat,
        acc_results, verbose, class_means, class_weights):
    """Runs a single pass over the dataset using matvecs.

    Args:
        dataset: A valid dataset object.
        kernel: A valid kernel object that can generate
            random features.
        q_mat (array): A (num_rffs, rank) array against
            which the random features will be multiplied.
        acc_results (array): A (num_rffs, rank) array
            in which Z^T Z @ q_mat will be stored.
        verbose (bool): Whether to print updates.
        class_means: Either None or a (nclasses, num_rffs)
            array storing the mean of the features for each
            class.
        class_weights: Either None or an (nclasses) array storing
            the class_weight for each class.
    """
    is_classification = class_means is not None

    for j, (xdata, ydata, ldata) in enumerate(dataset.get_chunked_data()):
        xdata, _ = kernel.transform_x_y(xdata, ydata, ldata,
                class_means, class_weights, is_classification)
        acc_results += xdata.T @ (xdata @ q_mat)
        if j % 10 == 0 and verbose:
            print(f"Chunk {j} complete.")


def single_pass_srht(dataset, kernel, compressor,
        acc_results, verbose, class_means, class_weights):
    """Runs a single pass over the dataset using SRHT.

    Args:
        dataset: A valid dataset object.
        kernel: A valid kernel object that can generate
            random features.
        compressor: A valid kernel object that can perform SRHT.
        acc_results (array): A (num_rffs, rank) array
            in which Z^T Z @ q_mat will be stored.
        verbose (bool): Whether to print updates.
        class_means: Either None or a (nclasses, num_rffs)
            array storing the mean of the features for each
            class.
        class_weights: Either None or an (nclasses) array storing
            the class_weight for each class.
    """
    is_classification = class_means is not None

    for j, (xdata, ydata, ldata) in enumerate(dataset.get_chunked_data()):
        xdata, _ = kernel.transform_x_y(xdata, ydata, ldata,
                class_means, class_weights, is_classification)
        acc_results += compressor.transform_x(xdata).T @ xdata
        if j % 10 == 0 and verbose:
            print(f"chunk {j} complete.")



def subsampled_srht(dataset, kernel, compressor, acc_results, verbose,
        sample_frac = 0.1, random_seed = 123, class_means=None,
        class_weights=None):
    """Runs a single pass over the dataset using SRHT, but sampling the
    data. The resulting preconditioner will not be useful for fitting
    but the calculated ratio is a good predictor of the number of
    iterations required to fit the full dataset, so it can determine
    what max_rank should be used to build the actual preconditioner
    for fitting.

    Args:
        dataset: A valid dataset object.
        kernel: A valid kernel object that can generate
            random features.
        compressor: A valid kernel object that can perform SRHT.
        acc_results (array): A (num_rffs, rank) array
            in which Z^T Z @ q_mat will be stored.
        verbose (bool): Whether to print updates.
        sample_frac (float): The fraction of datapoints to
            sample.
        random_seed (int): Seed for the random number generator.
        class_means: Either None or a (nclasses, num_rffs)
            array storing the mean of the features for each
            class.
        class_weights: Either None or an (nclasses) array storing
            the class_weight for each class.
    """
    rng = np.random.default_rng(random_seed)
    is_classification = class_means is not None

    for j, (xdata, ydata, ldata) in enumerate(dataset.get_chunked_data()):
        cutoff = max(int(sample_frac * float(xdata.shape[0])), 1)
        idx = rng.permutation(xdata.shape[0])[:cutoff]
        if ldata is not None:
            xdata, _ = kernel.transform_x_y(xdata[idx,...], ydata[idx],
                    ldata[idx], class_means, class_weights,
                    is_classification)
        else:
            xdata, _ = kernel.transform_x_y(xdata[idx,...], ydata[idx],
                    ldata, class_means, class_weights,
                    is_classification)
        acc_results += compressor.transform_x(xdata).T @ xdata
        if j % 10 == 0 and verbose:
            print(f"Chunk {j} complete.")


def single_pass_srht_zty(dataset, kernel, compressor, acc_results, z_trans_y,
                        verbose):
    """Runs a single pass over the dataset using SRHT and stores the
    z_trans_y vector. Note that this function should never be used
    for classification since it does not make sense to construct
    a z_trans_y vector in that case.

    Args:
        dataset: A valid dataset object.
        kernel: A valid kernel object that can generate
            random features.
        compressor: A valid kernel object that can perform SRHT.
        acc_results (array): A (num_rffs, rank) array
            in which Z^T Z @ q_mat will be stored.
        z_trans_y (array): A (num_rffs) array in which the
            product Z^T y will be stored.
        verbose (bool): Whether to print updates.
    """
    y_trans_y = 0.0
    for j, (xin, yin, ldata) in enumerate(dataset.get_chunked_data()):
        xdata, ydata = kernel.transform_x_y(xin, yin, ldata)
        z_trans_y += xdata.T @ ydata
        y_trans_y += ydata.T @ ydata
        acc_results += compressor.transform_x(xdata).T @ xdata
        if j % 10 == 0 and verbose:
            print(f"Chunk {j} complete.")

    return y_trans_y



def initialize_srht_multipass(dataset, rank, kernel, random_state, verbose = False,
                n_passes = 1, class_means=None, class_weights=None):
    """Builds the randomized Nystrom approximation to the inverse
    of (z^T z + lambda), where z is the random features generated
    for dataset, using SRHT on the first pass with subsequent passes over
    the dataset. This is more expensive but can result in a much
    better preconditioner under certain circumstances.

    Args:
        dataset: An OnlineDataset or OfflineDataset containing the raw data.
        rank (int): The desired rank of the preconditioner.
        kernel: A valid kernel object that can generate random features.
        random_state (int): A seed for the random number generator.
        verbose (bool): If True, print updates during construction.
        class_means: Either None or a (nclasses, num_rffs)
            array storing the mean of the features for each
            class.
        class_weights: Either None or an (nclasses) array storing
            the class_weight for each class.

    Returns:
        u_mat (np.ndarray): The eigenvectors of the matrix needed to
            form the preconditioner.
        s_mat (np.ndarray): The eigenvalues of the
            matrix needed to form the preconditioner.
        z_trans_y: Either None or an array containing the product Z^T @ y,
            depending on whether get_zty is True or False.
        y_trans_y: Either None or a float containing y^T y, depending on
            whether get_zty is True or False.
    """
    z_trans_y, y_trans_y = None, None
    if kernel.device == "cpu":
        acc_results = np.zeros((rank, kernel.get_num_rffs()))
        svd_calculator, cho_calculator = np.linalg.svd, np.linalg.cholesky
        tri_solver = scipy.linalg.solve_triangular
        qr_calculator = np.linalg.qr
        # If class means were supplied, we are doing classification and do
        # not need to retrieve z_trans_y. Otherwise we do need to retrieve
        # and store z_trans_y.
        if class_means is None:
            z_trans_y, y_trans_y = np.zeros((kernel.get_num_rffs())), 0.0
    else:
        mempool = cp.get_default_memory_pool()
        acc_results = cp.zeros((rank, kernel.get_num_rffs()))
        svd_calculator, cho_calculator = cp.linalg.svd, cp.linalg.cholesky
        tri_solver = cupyx.scipy.linalg.solve_triangular
        qr_calculator = cp.linalg.qr
        if class_means is None:
            z_trans_y, y_trans_y = cp.zeros((kernel.get_num_rffs())), 0.0

    compressor = SRHTCompressor(rank, kernel.get_num_rffs(),
                random_seed = random_state, device=kernel.device)

    # If class means were supplied, we are doing classification and do
    # not need to retrieve z_trans_y. Otherwise we do need to retrieve
    # and store z_trans_y.
    if class_means is None:
        y_trans_y = single_pass_srht_zty(dataset, kernel, compressor, acc_results,
                        z_trans_y, verbose)
    else:
        single_pass_srht(dataset, kernel, compressor, acc_results,
                verbose, class_means, class_weights)

    del compressor
    acc_results = acc_results.T

    if kernel.device == "cuda":
        mempool.free_all_blocks()

    for _ in range(n_passes - 1):
        q_mat, r_mat = qr_calculator(acc_results)
        acc_results[:] = 0.0
        del r_mat
        if kernel.device == "cuda":
            mempool.free_all_blocks()

        single_pass_gauss(dataset, kernel, q_mat, acc_results, verbose,
                class_means, class_weights)

    if kernel.device == "cuda":
        mempool.free_all_blocks()

    norm = float( np.sqrt((acc_results**2).sum())  )

    shift = np.spacing(norm)
    acc_results += shift * q_mat
    q_mat = q_mat.T @ acc_results

    q_mat = cho_calculator(q_mat)

    if kernel.device == "cuda":
        mempool.free_all_blocks()

    acc_results = tri_solver(q_mat, acc_results.T,
                            overwrite_b = True, lower=True).T

    u_mat, s_mat, _ = svd_calculator(acc_results, full_matrices=False)
    s_mat = (s_mat**2 - shift).clip(min=0)

    # Return z_trans_y if we are not doing classification.
    if class_means is None:
        return u_mat, s_mat, z_trans_y, y_trans_y
    return u_mat, s_mat, None, 0


def initialize_srht(dataset, rank, kernel, random_state, verbose = False,
                class_means=None, class_weights=None):
    """Builds the randomized Nystrom approximation to the inverse
    of (z^T z + lambda), where z is the random features generated
    for dataset, using SRHT.

    Args:
        dataset: An OnlineDataset or OfflineDataset containing the raw data.
        rank (int): The desired rank of the preconditioner.
        kernel: A valid kernel object that can generate random features.
        random_state (int): A seed for the random number generator.
        verbose (bool): If True, print updates during construction.
        class_means: Either None or a (nclasses, num_rffs)
            array storing the mean of the features for each
            class.
        class_weights: Either None or an (nclasses) array storing
            the class_weight for each class.

    Returns:
        u_mat (np.ndarray): The eigenvectors of the matrix needed to
            form the preconditioner.
        s_mat (np.ndarray): The eigenvalues of the
            matrix needed to form the preconditioner.
        z_trans_y: Either None or an array containing the product Z^T @ y,
            depending on whether get_zty is True or False.
        y_trans_y: Either None or a float containing y^T y, depending on
            whether get_zty is True or False.
    """
    z_trans_y, y_trans_y = None, None
    if kernel.device == "cpu":
        acc_results = np.zeros((rank, kernel.get_num_rffs()))
        svd_calculator, array_sqrt = np.linalg.svd, np.sqrt
        # If class means were supplied, we are doing classification and do
        # not need to retrieve z_trans_y. Otherwise we do need to retrieve
        # and store z_trans_y.
        if class_means is None:
            z_trans_y, y_trans_y = np.zeros((kernel.get_num_rffs())), 0.0
    else:
        mempool = cp.get_default_memory_pool()
        acc_results = cp.zeros((rank, kernel.get_num_rffs()))
        svd_calculator, array_sqrt = cp.linalg.svd, cp.sqrt
        if class_means is None:
            z_trans_y, y_trans_y = cp.zeros((kernel.get_num_rffs())), 0.0

    compressor = SRHTCompressor(rank, kernel.get_num_rffs(),
                random_seed = random_state, device=kernel.device)


    # If class means were supplied, we are doing classification and do
    # not need to retrieve z_trans_y. Otherwise we do need to retrieve
    # and store z_trans_y.
    if class_means is None:
        y_trans_y = single_pass_srht_zty(dataset, kernel, compressor, acc_results,
                        z_trans_y, verbose)
    else:
        single_pass_srht(dataset, kernel, compressor, acc_results, verbose,
                class_means, class_weights)

    c_mat = compressor.transform_x(acc_results)
    _, c_s1, c_v1 = svd_calculator(c_mat, full_matrices = False)

    del c_mat, compressor
    if kernel.device == "cuda":
        mempool.free_all_blocks()

    mask = c_s1 < 1e-14
    c_s1 = 1 / array_sqrt(c_s1.clip(min=1e-14))
    c_s1[mask] = 0
    acc_results = acc_results.T @ c_v1.T @ (c_s1[:,None] * c_v1)

    del c_v1, c_s1
    if kernel.device == "cuda":
        mempool.free_all_blocks()

    u_mat, s_mat, _ = svd_calculator(acc_results, full_matrices=False)
    s_mat = s_mat**2

    # Return z_trans_y if we are not doing classification.
    if class_means is None:
        return u_mat, s_mat, z_trans_y, y_trans_y
    return u_mat, s_mat, None, 0




def srht_ratio_check(dataset, rank, kernel, random_state, verbose = False,
                sample_frac = 0.1, class_means=None, class_weights=None):
    """Runs a fast 'preconditioner construction' using a random sample of
    the data. The resulting preconditioner will not be useful for fitting,
    so the eigenvectors normally needed for the preconditioner are not
    saved, but the eigenvalues can be used to get an estimated upper
    bound on the condition number of the Hessian formed from the sample,
    and this in turn can be used to estimate what max_rank is actually
    needed to build the preconditioner.

    Args:
        dataset: An OnlineDataset or OfflineDataset containing the raw data.
        rank (int): The desired rank of the preconditioner.
        kernel: A valid kernel object that can generate random features.
        random_state (int): A seed for the random number generator.
        verbose (bool): If True, print updates during construction.
        sample_frac (float): The fraction of datapoints to
            sample.
        class_means: Either None or a (nclasses, num_rffs)
            array storing the mean of the features for each
            class.
        class_weights: Either None or an (nclasses) array storing
            the class_weight for each class.

    Returns:
        s_mat (np.ndarray): The eigenvalues of the
            matrix needed for the ratio check.
    """
    if kernel.device == "cpu":
        acc_results = np.zeros((rank, kernel.get_num_rffs()))
        svd_calculator, array_sqrt = np.linalg.svd, np.sqrt
    else:
        mempool = cp.get_default_memory_pool()
        acc_results = cp.zeros((rank, kernel.get_num_rffs()))
        svd_calculator, array_sqrt = cp.linalg.svd, cp.sqrt

    compressor = SRHTCompressor(rank, kernel.get_num_rffs(),
                random_seed = random_state, device=kernel.device)

    # We never need to retrieve and store z_trans_y in this case.
    subsampled_srht(dataset, kernel, compressor, acc_results, verbose,
            sample_frac, random_state, class_means, class_weights)

    c_mat = compressor.transform_x(acc_results)
    _, c_s1, c_v1 = svd_calculator(c_mat, full_matrices = False)

    del c_mat, compressor
    if kernel.device == "cuda":
        mempool.free_all_blocks()

    mask = c_s1 < 1e-14
    c_s1 = 1 / array_sqrt(c_s1.clip(min=1e-14))
    c_s1[mask] = 0
    acc_results = acc_results.T @ c_v1.T @ (c_s1[:,None] * c_v1)

    del c_v1, c_s1
    if kernel.device == "cuda":
        mempool.free_all_blocks()

    _, s_mat, _ = svd_calculator(acc_results, full_matrices=False)
    s_mat = s_mat**2
    return s_mat
