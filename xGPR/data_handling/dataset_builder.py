"""Contains functions for building OnlineDataset and OfflineDataset objects.

When constructing a dataset, we should check to ensure the user has passed
valid data (to avoid cryptic errors during tuning or fitting). The functions
here provide the user with a way to construct online or offline (in memory
or on disk) datasets as appropriate while checking to ensure validity.
"""
import numpy as np

from .online_data_handling import OnlineDataset
from .offline_data_handling import OfflineDataset


def build_online_dataset(xdata, ydata, chunk_size = 2000):
    """build_online_dataset constructs an OnlineDataset
    object for data stored in memory, after first checking
    that some validity requirements are satisfied.

    Args:
        xdata (np.ndarray): A numpy array containing the x-values.
        ydata (np.ndarray): A numpy array containing the y-values.
        chunk_size (int): The maximum size of data chunks that
            will be returned to callers. Limits memory consumption.
            Defaults to 2000.

    Returns:
        dataset (OnlineDataset): An object of class OnlineDataset
            that can be passed to the hyperparameter tuning
            and fitting routines of the model classes.

    Raises:
        ValueError: If the data passed is not valid, the function
            will raise a detailed ValueError explaining the issue.
    """
    if not isinstance(xdata, np.ndarray) or not isinstance(ydata, np.ndarray):
        raise ValueError("X and y must be numpy arrays!")
    if len(ydata.shape) != 1:
        raise ValueError("Y must be a 1d numpy array.")
    if xdata.dtype != "float64" or ydata.dtype != "float64":
        raise ValueError("Both x and y must be arrays of datatype np.float64.")
    if ydata.shape[0] != xdata.shape[0]:
        raise ValueError("Different number of datapoints in x and y.")
    if np.sum(np.isnan(xdata)) > 0:
        raise ValueError("One or more elements in x is nan!")
    if np.max(xdata) > 1e15 or np.min(xdata) < -1e15:
        raise ValueError("Values > 1e15 or < -1e15 encountered. "
                    "Please rescale your data and check for np.inf.")

    dataset = OnlineDataset(xdata, ydata, chunk_size = chunk_size)
    return dataset


def build_offline_fixed_vector_dataset(xlist, ylist, chunk_size = 2000,
        skip_safety_checks = False):
    """Constructs an OfflineDataset
    object for data stored on disk, after first checking
    that some validity requirements are satisfied. This is
    intended for data that comes in fixed-vector form
    (e.g. tabular data or a sequence alignment).

    Args:
        xlist (list): A list of valid filepaths to .npy numpy
            arrays containing the x-data. These must all be 2d
            arrays where shape[1] is the number of features.
        ylist (list): A list of valid filepaths to .npy numpy
            arrays containing the y-data. These must all be 1d
            arrays. The number of datapoints in each should
            match the corresponding element of xlist.
        chunk_size (int): The maximum size of data chunks that
            will be returned to callers. Limits memory consumption.
            Defaults to 2000.
        skip_safety_checks (bool): If False, the builder will check
            each input array to make sure it does not contain
            infinite values or nan and that all the input arrays are 2d
            This is important -- if there are unexpected 3d arrays, np.nan,
            etc., this can lead to weird and unexpected results during
            fitting. On the other hand,
            this is a slow step and for a large dataset can take some
            time. Only skip_safety_checks if you have already checked your
            data and are confident that all is in order.

    Returns:
        dataset (OfflineDataset): An object of class OfflineDataset
            that can be passed to the hyperparameter tuning
            and fitting routines of the model classes.

    Raises:
        ValueError: If the data passed is not valid, the function
            will raise a detailed ValueError explaining the issue.
    """
    if len(xlist) == 0:
        raise ValueError("At least one datafile must be supplied.")
    xdim = [0,-1]
    if not skip_safety_checks:
        for xfile, yfile in zip(xlist, ylist):
            x_data = np.load(xfile)
            if x_data.shape[0] == 0:
                raise ValueError(f"File {xfile} has no datapoints.")
            if np.sum(np.isnan(x_data)) > 0:
                raise ValueError(f"One or more elements in file {xfile} is nan.")
            if np.max(x_data) > 1e15 or np.min(x_data) < -1e15:
                raise ValueError(f"One or more values in {xfile} is "
                        "> 1e15 or < -1e15. Please check for inf values "
                        "and / or rescale your data.")
            xdim[0] += x_data.shape[0]
            if xdim[1] == -1:
                xdim[1] = x_data.shape[1]
            elif x_data.shape[1] != xdim[1]:
                raise ValueError("All x arrays must have the same dimensionality.")

            yshape = _get_array_file_shape(yfile)
            if x_data.shape[0] != yshape[0]:
                raise ValueError(f"File {xfile} has a different number of datapoints "
                    f"than file {yfile}.")
            if len(x_data.shape) > 2:
                raise ValueError(f"File {xfile} is a > 2d array, but you have called "
                    "build_offline_2d_dataset.")
            if x_data.shape[0] > chunk_size:
                raise ValueError(f"Xfile {xfile} has more datapoints than allowed "
                    "based on specified chunk_size. Either increase chunk_size "
                    "or divide your data into np.ndarrays saved on disk that each "
                    "contain < chunk_size datapoints.")
            if len(yshape) > 1:
                raise ValueError(f"The y file {yfile} is not a 1d array.")

    #If we ARE skipping safety checks, retrieve the dimensionality info we will
    #need by checking files without loading them.
    else:
        xdim = [0,-1]
        for xfile in xlist:
            xshape = _get_array_file_shape(xfile)
            xdim[1] = xshape[1]
            xdim[0] += xshape[0]

    trainy_mean, trainy_std = _get_offline_scaling_factors(ylist)
    dataset = OfflineDataset(xlist, ylist, tuple(xdim),
                trainy_mean, trainy_std, chunk_size = chunk_size)
    return dataset


def build_offline_sequence_dataset(xlist, ylist, chunk_size = 2000,
        skip_safety_checks = False):
    """Constructs an OfflineDataset
    object for data stored on disk, after first checking
    that some validity requirements are satisfied. This is
    intended for data like time series or sequences.

    Args:
        xlist (list): A list of valid filepaths to .npy numpy
            arrays containing the x-data. These must all be 3d
            arrays where shape[1] is the number of timepoints
            or sequence elements and shape[2] is the number of
            features for each timepoint / sequence element.
        ylist (list): A list of valid filepaths to .npy numpy
            arrays containing the y-data. These must all be 1d
            arrays. The number of datapoints in each should
            match the corresponding element of xlist.
        chunk_size (int): The maximum size of data chunks that
            will be returned to callers. Limits memory consumption.
            Defaults to 2000.
        skip_safety_checks (bool): If False, the builder will check
            each input array to make sure it does not contain
            infinite values or nan and that all the input arrays are 3d
            This is important -- if there are unexpected 2d arrays, np.nan,
            etc., this can lead to weird and unexpected results during
            fitting. On the other hand,
            this is a slow step and for a large dataset can take some
            time. Only skip_safety_checks if you have already checked your
            data and are confident that all is in order.

    Returns:
        dataset (OfflineDataset): An object of class OfflineDataset
            that can be passed to the hyperparameter tuning
            and fitting routines of the model classes.

    Raises:
        ValueError: If the data passed is not valid, the function
            will raise a detailed ValueError explaining the issue.
    """
    if len(xlist) == 0:
        raise ValueError("At least one datafile must be supplied.")
    xdim = [0,-1,-1]
    if not skip_safety_checks:
        for xfile, yfile in zip(xlist, ylist):
            x_data = np.load(xfile)
            if x_data.shape[0] == 0:
                raise ValueError(f"File {xfile} has no datapoints.")
            if np.sum(np.isnan(x_data)) > 0:
                raise ValueError(f"One or more elements in file {xfile} is nan.")
            if np.max(x_data) > 1e15 or np.min(x_data) < -1e15:
                raise ValueError(f"One or more values in {xfile} is "
                    "> 1e15 or < -1e15. Please check for inf values "
                    "and / or rescale your data.")
            xdim[0] += x_data.shape[0]
            yshape = _get_array_file_shape(yfile)
            if x_data.shape[0] != yshape[0]:
                raise ValueError(f"File {xfile} has a different number of datapoints "
                    f"than file {yfile}.")
            if len(x_data.shape) != 3:
                raise ValueError(f"File {xfile} is not a 3d array, but you have called "
                "build_offline_3d_dataset.")

            if xdim[1] == -1:
                xdim[1] = x_data.shape[1]
                xdim[2] = x_data.shape[2]
            elif x_data.shape[1] != xdim[1] or x_data.shape[2] != xdim[2]:
                raise ValueError("All x arrays must have the same dimensionality.")
            if x_data.shape[0] > chunk_size:
                raise ValueError(f"Xfile {xfile} has more datapoints than allowed "
                    "based on specified chunk_size. Either increase chunk_size "
                    "or divide your data into np.ndarrays saved on disk that each "
                    "contain < chunk_size datapoints.")
            if len(yshape) > 1:
                raise ValueError(f"The y file {yfile} is not a 1d array.")

    #If we ARE skipping safety checks, retrieve the dimensionality info we will
    #need by checking files without loading them.
    else:
        xdim = [0,-1,-1]
        for xfile in xlist:
            xshape = _get_array_file_shape(xfile)
            xdim[1] = xshape[1]
            xdim[2] = xshape[2]
            xdim[0] += xshape[0]

    trainy_mean, trainy_std = _get_offline_scaling_factors(ylist)
    dataset = OfflineDataset(xlist, ylist, xdim,
                trainy_mean, trainy_std, chunk_size = chunk_size)
    return dataset


def _get_offline_scaling_factors(yfiles):
    """Gets scaling factors (mean and standard deviation)
    for 'offline' data stored on disk.

    Args:
        yfiles (list): A list of valid filepaths for
            .npy files (numpy arrays) containing the
            y-values for all datapoints.

    Returns:
        trainy_mean (float): The mean of the training y-data.
        trainy_std (float): The standard deviation of the
            training y-data.
    """
    ndpoints = 0
    trainy_mean, trainy_var = 0.0, 0.0
    for yfile in yfiles:
        y_data = np.load(yfile).astype(np.float64)
        w_1 = y_data.shape[0] / (y_data.shape[0] + ndpoints)
        w_2 = ndpoints / (ndpoints + y_data.shape[0])
        w_3 = y_data.shape[0] * ndpoints / (y_data.shape[0] + ndpoints)**2
        y_data_mean = y_data.mean()

        trainy_var = w_1 * y_data.std()**2 + w_2 * trainy_var + w_3 * (y_data_mean -
                trainy_mean)**2
        trainy_mean = w_1 * y_data_mean + w_2 * trainy_mean
        ndpoints += y_data.shape[0]

    trainy_std = np.sqrt(trainy_var)
    return trainy_mean, trainy_std


def _get_array_file_shape(npy_file):
    """Gets the shape of a .npy file array without loading it
    to memory."""
    with open(npy_file, 'rb') as f_handle:
        major, _ = np.lib.format.read_magic(f_handle)
        if major == 1:
            arr_shape, _, _ = np.lib.format.read_array_header_1_0(f_handle)
        else:
            arr_shape, _, _ = np.lib.format.read_array_header_2_0(f_handle)
    return arr_shape
