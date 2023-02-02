"""The baseclass for all other standard kernels and for SORFKernelBaseclass.

KernelBaseclass houses methods and attributes used by all kernels, including
for resetting the device, calculating the gradient for hyperparameters that
all kernels have etc. It also includes some abstractmethods to ensure
each kernel exposes the same methods to the model classes.
"""
import abc
from abc import ABC

import numpy as np
try:
    import cupy as cp
except:
    pass


class KernelBaseclass(ABC):
    """The baseclass for all other kernel classes.

    Attributes:
        num_freqs (int): The number of sampled frequencies.
        num_rffs (int): The number of random fourier features. Note that
            this will be the same for most kernels. The exception is "sine-
            cosine kernels", e.g. RBF, Matern, which use both the sine
            and cosine of each feature to generate two separate random
            features. In this case the user-requested num_rffs must
            be an even number and num_rffs will be set equal to this
            value, while num_freqs will be 0.5 * num_features.
        xdim (tuple): The dimensionality of the input. Only elements
            1 and onwards are used -- element 0 is the number of datapoints
            and is not used by the kernel, so any value for element 0
            is acceptable.
        device (str): Must be one of 'cpu', 'gpu'. Determines where
            calculations are performed.
        hyperparams (np.ndarray): An array of shape (N) for N hyperparameters.
            Initialized here to None, each child class will set to its
            own defaults.
        bounds (np.ndarray): An array of shape (N, 2) for N hyperparameters
            that determines the optimization bounds for hyperparameter tuning.
            Each kernel has its own defaults. The parent class initializes to
            None, child classes must set this value.
        zero_arr: A reference to either np.zeros or cp.zeros, depending on
            self.device. This is for convenience and ensures that child
            classes can call self.zero_arr and get an array appropriate
            for the current device.
        dtype: A reference to either np.float64, np.float32, cp.float32, cp.float64
            depending on self.device and self.double_precision (see below).
        out_type: A reference to either np.float64 or cp.float64 depending on
            device.
        empty: A reference to either np.emtpy or cp.empty depending on self.device.
        double_precision (bool): If True, generate random features in double precision.
            Otherwise, generate as single precision.
    """

    def __init__(self, num_rffs, xdim, double_precision = True,
            sine_cosine_kernel = False):
        """Constructor for the KernelBaseclass.

        Args:
            num_rffs (int): The user-requested number of random Fourier features.
                For sine-cosine kernels (RBF, Matern), this will be saved by the
                class as num_rffs.
            xdim (tuple): The dimensions of the input. Either (N, D) or (N, M, D)
                where N is the number of datapoints, D is number of features
                and M is number of timepoints or sequence elements (convolution
                kernels only).
            double_precision (bool): If True, generate random features in double precision.
                Otherwise, generate as single precision.
            sine_cosine_kernel (bool): If True, the kernel is a sine-cosine kernel,
                meaning it will sample self.num_freqs frequencies and use the sine
                and cosine of each to generate twice as many features
                (self.num_rffs). sine-cosine kernels only accept num_rffs
                that are even numbers.

        Raises:
            ValueError: Raises a ValueError if a sine-cosine kernel is requested
                but num_rffs is not an integer multiple of 2.
        """
        self.double_precision = double_precision
        if sine_cosine_kernel:
            if num_rffs <= 1 or not (num_rffs / 2).is_integer():
                raise ValueError("For sine-cosine kernels (e.g. matern, rbf) "
                        "the number of random fourier features must be an integer "
                        "multiple of two.")
            self.num_freqs = int(num_rffs / 2)
            self.num_rffs = num_rffs
        else:
            self.num_freqs = num_rffs
            self.num_rffs = num_rffs
        self.xdim = xdim
        self.hyperparams = None
        self.bounds = None



    @abc.abstractmethod
    def transform_x(self, input_x):
        """Kernel classes must implement a method that generates
        random features for a given set of inputs."""

    @abc.abstractmethod
    def kernel_specific_set_device(self, new_device):
        """Kernel classes must implement a method that performs
        any kernel-specific operations needed to switch device
        for the kernel."""


    def check_bounds(self, bounds):
        """Checks a set of bounds provided by the caller to ensure they
        are valid for the selected kernel.

        Args:
            bounds (np.ndarray): A numpy array of shape (N, 2) where N
                is the number of hyperparameters, with bounds as
                (low, high).

        Raises:
            ValueError: A ValueError is raised if the bounds array
                passed is not valid for this kernel.
        """
        if bounds.shape != self.bounds.shape:
            raise ValueError("You have tried to supply a set of bounds "
                "for hyperparameter tuning that do not match the shape "
                "of the bounds for the kernel you have chosen. The bounds "
                "should be a numpy array of shape [n_hyperparams, 2] "
                "where the first column is the low bound, second column "
                "is the high bound.")


    def get_bounds(self, logspace=True):
        """Returns the bounds for hyperparameter optimization.
        Note that we do not make this a property because callers
        may need to specify whether they are using logspace (or not).

        Args:
            logspace (bool): If True, return the log of the boundary
                values, otherwise, return the actual values.

        Returns:
            self.bounds (np.ndarray): A numpy array of shape
                (N, 2) where N is the number of hyperparameters,
                containing either the (low, high) bounds
                or the natural lof of those bounds.
        """
        if logspace:
            return np.log(self.bounds)
        return self.bounds


    def set_bounds(self, bounds, logspace=True):
        """Sets the bounds for hyperparameter optimization.
        Note that we do not make this a property because callers
        may need to specify whether they are using logspace (or not).

        Args:
            bounds (np.ndarray): Must be an array of shape (N, 2) where
                N is the number of hyperparameters, with bounds as
                (low, high).
            logspace (bool): If True, the supplied values are the log
                of the actual bounds.

        Raises:
            ValueError: A ValueError is raised if the bounds array
                passed by caller is invalid for the chosen kernel.
        """
        self.check_bounds(bounds)
        if logspace:
            self.bounds = np.exp(bounds)
        else:
            self.bounds = bounds


    def validate_new_datapoints(self, input_x):
        """Checks new input data supplied by caller to
        ensure it is compatible with the dimensionality
        of the data used to fit the model.

        Args:
            input_x (np.ndarray): A numpy array containing
                raw input data.

        Returns:
            valid_data (bool): True if input was acceptable,
                False otherwise.
        """
        valid_data = True
        if len(input_x.shape) != len(self.xdim):
            valid_data = False
        if len(self.xdim) == 3:
            if input_x.shape[2] != self.xdim[2]:
                valid_data = False
        if input_x.shape[1] != self.xdim[1]:
            valid_data = False
        return valid_data



    def get_hyperparams(self, logspace = True):
        """Returns the kernel hyperparameters. We have not used the property
        decorator since the logspace option is required.

        Args:
            logspace (bool): If True, the natural log of the hyperparameters is
                returned. Defaults to True.

        Returns:
            hyperparameters (np.ndarray): A numpy array of shape (N) for N
                kernel hyperparameters.
        """
        if logspace:
            return np.log(self.hyperparams)
        return self.hyperparams



    def set_hyperparams(self, hyperparams, logspace = True):
        """Sets the kernel hyperparameters. We have not used the property
        decorator since the logspace option is required.

        Args:
            hyperparams (np.ndarray): A numpy array of shape N for N hyperparameters.
                This function does not check for validity since it is often
                used by optimization algorithms that are merely modifying
                hyperparameters they retrieved from get_hyperparams. A caller
                that is passing values which may be invalid should call
                self.check_hyperparams first.
            logspace (bool): If True, the natural log of the hyperparameters is
                returned. Defaults to True.

        Returns:
            hyperparameters (np.ndarray): A numpy array of shape (N) for N
                kernel hyperparameters.
        """
        if logspace:
            self.hyperparams = np.exp(hyperparams)
        else:
            self.hyperparams = hyperparams


    def get_lambda(self):
        """For convenience, we enable caller to retrieve only the
        first hyperparameter, which is needed for a variety of
        operations during fitting and tuning. This hyperparameter
        determines the 'noise level' of the data."""
        return self.hyperparams[0]


    def get_beta(self):
        """For convenience, we enable caller to retrieve only the
        second hyperparameter, which is needed for a variety of
        operations during fitting and tuning."""
        return self.hyperparams[1]


    def check_hyperparams(self, hyperparams):
        """Checks a suggested set of hyperparameters passed
        by caller to ensure they are valid. This should be used
        when callers want to set the hyperparameters to user-specified
        values. The optimization algorithms bypass this because
        they are merely modifying hyperparameters retrieved from
        the kernel.

        Args:
            hyperparams (np.ndarray): A numpy array of shape (N), where
                N is the number of hyperparameters.

        Raises:
            ValueError: Raises a ValueError if invalid hyperparameters are
                passed.
        """
        if not isinstance(hyperparams, np.ndarray):
            raise ValueError("The starting hyperparameters must be a numpy array.")
        if hyperparams.shape != self.hyperparams.shape:
            raise ValueError(f"The kernel you selected uses {self.hyperparams.shape[0]} "
                "hyperparameters. A hyperparameter array of the incorrect shape was passed.")



    def get_num_rffs(self):
        """Returns number of RFFs. Not a @property because
        we do not want to define a setter, external classes
        should not be able to set."""
        return self.num_rffs


    def set_out_type(self, out_type = "f"):
        """Temporarily changes the out data type, which will be
        reset whenever the device is changed. This is convenient
        for certain operations (e.g. pretransforming data)
        which do not require double precision."""
        if out_type == "f":
            if self.device == "gpu":
                self.out_type = cp.float32
            else:
                self.out_type = np.float32
        elif out_type == "d":
            if self.device == "gpu":
                self.out_type = cp.float64
            else:
                self.out_type = np.float64
        else:
            raise ValueError("Unrecognized data type supplied to kernel.")


    @property
    def device(self):
        """Getter for the device property, which determines
        whether calculations are on CPU or GPU."""
        return self.device_


    @device.setter
    def device(self, value):
        """Setter for device, which determines whether calculations
        are on CPU or GPU. Note that each kernel must also have
        a kernel_specific_set_device function (enforced via
        an abstractmethod) to make any kernel-specific changes
        that occur when the device is switched.

        Args:
            value (str): Must be one of 'cpu', 'gpu'.

        Raises:
            ValueError: A ValueError is raised if an unrecognized
                device is passed.

        Note that a number of 'convenience attributes' (e.g. self.dtype,
        self.zero_arr) are set as references to either cupy or numpy functions.
        This avoids having to write two sets of functions (one for cupy, one for
        numpy) for each gradient calculation when the steps involved are the same.
        Also note that cupy uses float32, which is 5-10x faster on GPU; on CPU,
        float32 provides a much more modest benefit and float64 is used instead.
        """
        if value == "cpu":
            self.empty = np.empty
            self.zero_arr = np.zeros
            self.out_type = np.float64
            if self.double_precision:
                self.dtype = np.float64
            else:
                self.dtype = np.float32

        elif value == "gpu":
            self.empty = cp.empty
            self.zero_arr = cp.zeros
            self.out_type = cp.float64
            if self.double_precision:
                self.dtype = cp.float64
            else:
                self.dtype = cp.float32
        else:
            raise ValueError("Unrecognized device supplied. Must be one "
                    "of 'cpu', 'gpu'.")
        self.device_ = value
        self.kernel_specific_set_device(value)
