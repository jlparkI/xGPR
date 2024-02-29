"""Contains functions for building OnlineDataset and OfflineDataset objects.

When constructing a dataset, we should check to ensure the user has passed
valid data (to avoid cryptic errors during tuning or fitting). The functions
here provide the user with a way to construct online or offline (in memory
or on disk) datasets as appropriate while checking to ensure validity.
"""
import numpy as np

from .online_data_handling import OnlineDataset
from .offline_data_handling import OfflineDataset


def build_regression_dataset(xdata, ydata, sequence_lengths = None,
        chunk_size:int = 2000, normalize_y:bool = True):
    """Builds a dataset intended for use for regression.

    Args:
        xdata (np.ndarray): Either a numpy array containing the x-values
            or a list of valid filepaths to .npy numpy arrays containing
            the x-data. If the latter, these must all be either
            2d arrays where shape[1] is the number of features or
            3d arrays where shape[1] is the number of timepoints
            or sequence elements and shape[2] is the number of
            features for each timepoint / sequence element. They
            should be one or the other, not both.
        ydata (np.ndarray): Either a numpy array containing the y-values
            or a list of valid filepaths to .npy numpy
            arrays containing the y-data. These must all be 1d
            arrays. The number of datapoints in each should
            match the corresponding element of xlist.
        sequence_lengths (np.ndarray): Either (a) a numpy array containing
            the length of each sequence / timeseries / graph for each datapoint or
            (b) a list of valid filepaths to .npy files containing the
            length of each sequence / timeseries / graph or (c) None. If you are
            not using a sequence kernel, this must always be None. If you are
            using a sequence / graph kernel, this cannot be None. In other words,
            this is required if using a sequence / graph kernel and must be
            None otherwise.
        chunk_size (int): The maximum size of data chunks that
            will be returned to callers. Limits memory consumption.
            Defaults to 2000. If working with files on disk, the
            files will be checked to ensure that none of them is
            > than this size.
        normalize_y (bool): If True, y values are normalized. Generally a
            good idea, unless you have already selected hyperparameters based
            on prior knowledge.

    Returns:
        dataset: An object of class OnlineDataset or OfflineDataset
            that can be passed to the hyperparameter tuning
            and fitting routines of the model classes.

    Raises:
        ValueError: A ValueError is raised if inappropriate argument
            types are supplied.
    """
    if isinstance(xdata, list) and isinstance(ydata, list):
        return build_offline_np_dataset(xdata, ydata, sequence_lengths,
                chunk_size, normalize_y, task_type = "regression")
    if isinstance(xdata, np.ndarray) and isinstance(ydata, np.ndarray):
        return build_online_dataset(xdata, ydata, sequence_lengths,
                chunk_size, normalize_y, task_type = "regression")
    raise ValueError("Unexpected argument types to build_regression_dataset.")



def build_classification_dataset(xdata, ydata,
        sequence_lengths = None, chunk_size:int = 2000):
    """Builds a dataset intended for use for classification.

    Args:
        xdata (np.ndarray): Either a numpy array containing the x-values
            or a list of valid filepaths to .npy numpy arrays containing
            the x-data. If the latter, these must all be either
            2d arrays where shape[1] is the number of features or
            3d arrays where shape[1] is the number of timepoints
            or sequence elements and shape[2] is the number of
            features for each timepoint / sequence element. They
            should be one or the other, not both.
        ydata (np.ndarray): Either a numpy array containing the y-values
            or a list of valid filepaths to .npy numpy
            arrays containing the y-data. These must all be 1d
            arrays. The number of datapoints in each should
            match the corresponding element of xlist. The category
            for each datapoint should be specified as an integer in
            [0, max category]. E.g. if there are three categories
            they will be numbereed 0, 1, 2.
        sequence_lengths (np.ndarray): Either (a) a numpy array containing
            the length of each sequence / timeseries / graph for each datapoint or
            (b) a list of valid filepaths to .npy files containing the
            length of each sequence / timeseries / graph or (c) None. If you are
            not using a sequence kernel, this must always be None. If you are
            using a sequence / graph kernel, this cannot be None. In other words,
            this is required if using a sequence / graph kernel and must be
            None otherwise.
        chunk_size (int): The maximum size of data chunks that
            will be returned to callers. Limits memory consumption.
            Defaults to 2000. If working with files on disk, the
            files will be checked to ensure that none of them is
            > than this size.

    Returns:
        dataset: An object of class OnlineDataset or OfflineDataset
            that can be passed to the hyperparameter tuning
            and fitting routines of the model classes.

    Raises:
        ValueError: A ValueError is raised if inappropriate argument
            types are supplied.
    """
    if isinstance(xdata, list) and isinstance(ydata, list):
        return build_offline_np_dataset(xdata, ydata, sequence_lengths,
                chunk_size, normalize_y = False, task_type = "classification")
    if isinstance(xdata, np.ndarray) and isinstance(ydata, np.ndarray):
        return build_online_dataset(xdata, ydata, sequence_lengths,
                chunk_size, normalize_y = False, task_type = "classification")
    raise ValueError("Unexpected argument types to build_regression_dataset.")




def build_online_dataset(xdata, ydata, sequence_lengths = None,
        chunk_size:int = 2000, normalize_y:bool = True,
        task_type:str = "regression"):
    """build_online_dataset constructs an OnlineDataset
    object for data stored in memory, after first checking
    that some validity requirements are satisfied.

    Args:
        xdata (np.ndarray): A numpy array containing the x-values.
        ydata (np.ndarray): A numpy array containing the y-values.
        sequence_lengths (np.ndarray): Either None or a numpy array containing
            the length of each sequence / timeseries / graph for each datapoint.
            If you are not using a sequence kernel, this must always be None. If you are
            using a sequence / graph kernel, this cannot be None.
        chunk_size (int): The maximum size of data chunks that
            will be returned to callers. Limits memory consumption.
            Defaults to 2000.
        normalize_y (bool): If True, y values are normalized. Generally a
            good idea, unless you have already selected hyperparameters based
            on prior knowledge.
        task_type (str): One of "regression", "classification". Indicates
            what purpose the dataset will be used for.

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

    __check_sequence_length(sequence_lengths, xdata, ydata)

    if len(ydata.shape) != 1:
        raise ValueError("Y must be a 1d numpy array.")
    if xdata.dtype not in ("float64", "float32"):
        raise ValueError("x must be an array of type float32 or type float64.")
    if ydata.dtype != "float64" and task_type == "regression":
        raise ValueError("For regression, ydata must be an array of type float64.")
    if task_type == "classification" and not issubclass(ydata.dtype.type,
                np.integer):
        raise ValueError("For classification, ydata must be an array of integers.")

    if ydata.shape[0] != xdata.shape[0]:
        raise ValueError("Different number of datapoints in x and y.")
    if np.isnan(xdata).any():
        raise ValueError("One or more elements in x is nan!")
    if np.max(xdata) > 1e15 or np.min(xdata) < -1e15:
        raise ValueError("Values > 1e15 or < -1e15 encountered. "
                    "Please rescale your data and check for np.inf.")

    if task_type == "regression":
        if normalize_y:
            dataset = OnlineDataset(xdata, ydata, sequence_lengths,
                chunk_size = chunk_size, trainy_mean = ydata.mean(),
                trainy_std = ydata.std())
        else:
            dataset = OnlineDataset(xdata, ydata, sequence_lengths,
                    chunk_size = chunk_size)

    else:
        dataset = OnlineDataset(xdata, ydata, sequence_lengths,
                chunk_size = chunk_size, max_class = ydata.max())
        if ydata.min() != 0:
            raise ValueError("For classification, there must be a zero category.")

    return dataset


def build_offline_np_dataset(xlist:list, ylist:list, sequence_lengths,
        chunk_size:int = 2000, normalize_y:bool = True,
        skip_safety_checks:bool = False, task_type:str = "regression"):
    """Constructs an OfflineDataset for data stored on disk
    as a list of npy files, after checking validity requirements.

    Args:
        xlist (list): A list of valid filepaths to .npy numpy
            arrays containing the x-data. These must all be either
            2d arrays where shape[1] is the number of features or
            3d arrays where shape[1] is the number of timepoints
            or sequence elements and shape[2] is the number of
            features for each timepoint / sequence element. They
            should be one or the other, not both.
        ylist (list): A list of valid filepaths to .npy numpy
            arrays containing the y-data. These must all be 1d
            arrays. The number of datapoints in each should
            match the corresponding element of xlist.
        sequence_lengths (np.ndarray): Either None or a list of valid filepaths to
            .npy files containing the length of each sequence / timeseries / graph. If
            not using a sequence kernel, this must always be None. If you are
            using a sequence / graph kernel, this cannot be None.
        chunk_size (int): The maximum size of data chunks that
            will be returned to callers. Limits memory consumption.
            Defaults to 2000.
        normalize_y (bool): If True, y values are normalized. Generally a
            good idea, unless you have already selected hyperparameters based
            on prior knowledge.
        skip_safety_checks (bool): If False, the builder will check
            each input array to make sure it does not contain
            infinite values or nan and that all the input arrays are 2d
            This is important -- if there are unexpected 3d arrays, np.nan,
            etc., this can lead to weird and unexpected results during
            fitting. On the other hand,
            this is a slow step and for a large dataset can take some
            time. Only skip_safety_checks if you have already checked your
            data and are confident that all is in order.
        task_type (str): One of "regression", "classification". Indicates
            what purpose the dataset will be used for.

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
    if len(xlist) != len(ylist):
        raise ValueError("xlist and ylist must have the same length.")
    if not isinstance(xlist, list) or not isinstance(ylist, list):
        raise ValueError("Both xlist and ylist should be lists.")

    if sequence_lengths is None:
        length_files = [None for y in ylist]
    else:
        length_files = sequence_lengths
        if len(sequence_lengths) != len(ylist):
            raise ValueError("sequence_lengths must either be None or "
                    "have the same length as ylist.")

    xshape = _get_array_file_shape(xlist[0])
    expected_arrlen = len(xshape)
    if expected_arrlen == 2:
        xdim = [0,-1]
    elif expected_arrlen == 3:
        xdim = [0,-1,-1]
    else:
        raise ValueError("Arrays should be either 2d or 3d.")

    if not skip_safety_checks:
        for xfile, yfile, length_file in zip(xlist, ylist, length_files):
            x_data = np.load(xfile)
            if x_data.shape[0] == 0:
                raise ValueError(f"File {xfile} has no datapoints.")
            if np.isnan(x_data).any():
                raise ValueError(f"One or more elements in file {xfile} is nan.")
            if np.max(x_data) > 1e15 or np.min(x_data) < -1e15:
                raise ValueError(f"One or more values in {xfile} is "
                        "> 1e15 or < -1e15. Please check for inf values "
                        "and / or rescale your data.")
            if len(x_data.shape) != expected_arrlen:
                raise ValueError(f"File {xfile} is not a {expected_arrlen}d "
                    "array, unlike some other arrays in xlist.")

            xdim[0] += x_data.shape[0]
            if xdim[1] == -1:
                xdim[1] = x_data.shape[1]
            if expected_arrlen == 2:
                if x_data.shape[1] != xdim[1]:
                    raise ValueError("All x arrays must have the same dimensionality.")
            elif expected_arrlen == 3:
                if xdim[2] == -1:
                    xdim[2] = x_data.shape[2]
                elif x_data.shape[2] != xdim[2]:
                    raise ValueError("All x arrays must have the same dimensionality.")

            ydata = np.load(yfile)
            if x_data.shape[0] != ydata.shape[0]:
                raise ValueError(f"File {xfile} has a different number of datapoints "
                    f"than file {yfile}.")
            if x_data.shape[0] > chunk_size:
                raise ValueError(f"Xfile {xfile} has more datapoints than allowed "
                    "based on specified chunk_size. Either increase chunk_size "
                    "or divide your data into np.ndarrays saved on disk that each "
                    "contain < chunk_size datapoints.")
            if len(ydata.shape) > 1:
                raise ValueError(f"The y file {yfile} is not a 1d array.")

            if length_file is None:
                __check_sequence_length(length_file, x_data, ydata)
            else:
                __check_sequence_length(np.load(length_file), x_data, ydata)


    #If we ARE skipping safety checks, retrieve the dimensionality info we will
    #need by checking files without loading them.
    else:
        xshape = _get_array_file_shape(xlist[0])
        xdim[1] = xshape[1]
        xdim[0] += xshape[0]
        if expected_arrlen == 3:
            xdim[2] = xshape[2]

    if normalize_y and task_type == "regression":
        max_class = 1
        trainy_mean, trainy_std = _get_offline_scaling_factors(ylist)
    else:
        trainy_mean, trainy_std = 0.0, 1.0
    if task_type == "classification":
        max_class, class_data_err = _get_offline_ymax(ylist)
        if class_data_err:
            raise ValueError("For classification, there must be a zero category, "
                "and all yfiles must be integers.")

    dataset = OfflineDataset(xlist, ylist, sequence_lengths, tuple(xdim),
                trainy_mean = trainy_mean, trainy_std = trainy_std,
                max_class = max_class, chunk_size = chunk_size)
    return dataset



def __check_sequence_length(seqlength, xdata, ydata):
    """Check a sequence length array to make sure it complies
    with typical requirements.

    Args:
        seqlength: Either None or a numpy array.
        ydata (np.ndarray): A numpy array.
        xdata (np.ndarray): A numpy array.

    Raises:
        ValueError: A ValueError is raised if requirements are
            not met. Otherwise if nothing is returned everything is ok.
    """
    if seqlength is not None:
        if len(xdata.shape) != 3:
            raise ValueError("sequence_length must be None if using "
                    "fixed vector input.")
        if not isinstance(seqlength, np.ndarray):
            raise ValueError("sequence_length must either be None or "
                    "a numpy array.")
        if not len(seqlength.shape) == 1:
            raise ValueError("sequence_length, if not None, must be a "
                    "1d numpy array.")
        if seqlength.shape[0] != ydata.shape[0]:
            raise ValueError("sequence_length, if not None, must have "
                    "the same length as ydata.")
        if not issubclass(seqlength.dtype.type, np.integer):
            raise ValueError("sequence_length, if not None, must be all "
                    "integers.")
        if seqlength.min() <= 0 or seqlength.max() > xdata.shape[1]:
            raise ValueError("sequence_length values must be in the range "
                    "(1, num_elements) where num_elements is the number of "
                    "sequence elements / graph nodes in the corresponding "
                    "input array.")
    elif len(xdata.shape) == 3:
        raise ValueError("sequence_lengths cannot be None if supplying "
                "sequences / time series / graphs as input.")


def _get_offline_ymax(yfiles:list):
    """Gets the max category for 'offline' data stored on disk.

    Args:
        yfiles (list): A list of valid filepaths for
            .npy files (numpy arrays) containing the
            y-values for all datapoints.

    Returns:
        max_class (int): The maximum category found in the data.
        class_data_err (bool): If True, one or more of the files
            has an incorrect format (noninteger data) or there
            is no zero category.
    """
    max_class, min_class, class_data_err = 0, 1, False
    for yfile in yfiles:
        y_data = np.load(yfile)
        if not issubclass(y_data.dtype.type, np.integer):
            class_data_err = True
            break
        if y_data.max() > max_class:
            max_class = y_data.max()
        if y_data.min() < min_class:
            min_class = y_data.min()

    if max_class == 0 or min_class != 0:
        class_data_err = True
    return max_class, class_data_err




def _get_offline_scaling_factors(yfiles:list):
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


def _get_array_file_shape(npy_file:str):
    """Gets the shape of a .npy file array without loading it
    to memory."""
    with open(npy_file, 'rb') as f_handle:
        major, _ = np.lib.format.read_magic(f_handle)
        if major == 1:
            arr_shape, _, _ = np.lib.format.read_array_header_1_0(f_handle)
        else:
            arr_shape, _, _ = np.lib.format.read_array_header_2_0(f_handle)
    return arr_shape
