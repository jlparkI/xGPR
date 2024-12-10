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
        _xdim (tuple): The dimensionality of the input. Only elements
            1 and onwards are used -- element 0 is the number of datapoints
            and is not used by the kernel, so any value for element 0
            is acceptable.
        device (str): Must be one of 'cpu', 'cuda'. Determines where
            calculations are performed.
        hyperparams (np.ndarray): An array of shape (N) for N hyperparameters.
            Initialized here to None, each child class will set to its
            own defaults.
        bounds (np.ndarray): An array of shape (N, 2) for N hyperparameters
            that determines the optimization bounds for hyperparameter tuning.
            Each kernel has its own defaults. The parent class initializes to
            None, child classes must set this value.
        double_precision (bool): If True, generate random features in double precision.
            Otherwise, generate as single precision.
        fit_intercept (bool): Whether to fit a y-intercept. Defaults to True but can
            be set to False by adding "intercept":False to kernel_spec_parms.
    """

    def __init__(self, num_rffs, xdim, num_threads = 2,
            sine_cosine_kernel = False, double_precision = False,
            kernel_spec_parms = {}):
        """Constructor for the KernelBaseclass.

        Args:
            num_rffs (int): The user-requested number of random Fourier features.
                For sine-cosine kernels (RBF, Matern), this will be saved by the
                class as num_rffs.
            xdim (tuple): The dimensions of the input. Either (N, D) or (N, M, D)
                where N is the number of datapoints, D is number of features
                and M is number of timepoints or sequence elements (convolution
                kernels only).
            num_threads (int): The number of threads to use if running on CPU. If
                running on GPU, this is ignored.
            sine_cosine_kernel (bool): If True, the kernel is a sine-cosine kernel,
                meaning it will sample self.num_freqs frequencies and use the sine
                and cosine of each to generate twice as many features
                (self.num_rffs). sine-cosine kernels only accept num_rffs
                that are even numbers.
            double_precision (bool): If True, generate random features in double precision.
                Otherwise, generate as single precision.
            kernel_spec_parms (dict): A dictionary of kernel settings / kernel-specific
                parameters.

        Raises:
            ValueError: Raises a ValueError if a sine-cosine kernel is requested
                but num_rffs is not an integer multiple of 2.
        """
        self.double_precision = double_precision
        if num_rffs < 2:
            raise ValueError("num_rffs should always be >= 2.")

        if sine_cosine_kernel:
            if not (num_rffs / 2).is_integer():
                raise ValueError("For sine-cosine kernels (e.g. matern, rbf) "
                        "the number of random fourier features must be an integer "
                        "multiple of two.")
            self.num_freqs = int(num_rffs / 2)
            self.num_rffs = num_rffs
        else:
            self.num_freqs = num_rffs
            self.num_rffs = num_rffs

        self.fit_intercept = True
        if "intercept" in kernel_spec_parms:
            if kernel_spec_parms["intercept"] is False:
                self.fit_intercept = False

        self._xdim = xdim
        self.hyperparams = None
        self.bounds = None

        self.num_threads = num_threads



    @abc.abstractmethod
    def kernel_specific_transform(self, input_x, sequence_length):
        """Kernel classes must implement a method that generates
        random features for a given set of inputs."""

    @abc.abstractmethod
    def kernel_specific_gradient(self, input_x, sequence_length):
        """Kernel classes must implement a method that calculates
        the NMLL gradient for any kernel-specific hyperparameters."""

    @abc.abstractmethod
    def kernel_specific_set_device(self, new_device):
        """Kernel classes must implement a method that performs
        any kernel-specific operations needed to switch device
        for the kernel."""

    @abc.abstractmethod
    def kernel_specific_set_hyperparams(self):
        """Kernel classes must implement a method that performs
        any kernel-specific changes necessary after the hyperparameters
        have been reset."""


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
        if len(input_x.shape) != len(self._xdim):
            valid_data = False
        elif len(self._xdim) == 3:
            if input_x.shape[2] != self._xdim[2]:
                valid_data = False
            if input_x.shape[1] < 1:
                valid_data = False
        elif input_x.shape[1] != self._xdim[1]:
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
        """Sets the kernel hyperparameters.

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
        self.kernel_specific_set_hyperparams()


    def get_lambda(self):
        """For convenience, we enable caller to retrieve only the
        first hyperparameter, which is needed for a variety of
        operations during fitting and tuning. This hyperparameter
        determines the 'noise level' of the data."""
        return self.hyperparams[0]



    def transform_x(self, input_x, sequence_length = None):
        """Given a numpy array as input (and sequence_length,
        which is none for most kernels but must be specified
        for convolution kernels), generate random features
        as output."""
        # This always generates a copy, which means that we
        # are never working on the input data, only on a copy,
        # and can therefore modify it with impunity.
        if not input_x.flags["C_CONTIGUOUS"]:
            if self.double_precision:
                xin = np.ascontiguousarray(input_x, np.float64)
            else:
                xin = np.ascontiguousarray(input_x, np.float32)
        elif self.double_precision:
            xin = input_x.astype(np.float64, copy=True)
        else:
            xin = input_x.astype(np.float32, copy=True)

        if self.device == "cuda":
            xin = cp.asarray(xin)

        slen = None
        if sequence_length is not None:
            slen = sequence_length.astype(np.int32, copy=False)

        xtrans = self.kernel_specific_transform(xin, slen)
        if self.fit_intercept:
            xtrans[:,0] = 1.

        return xtrans



    def transform_x_y(self, input_x, input_y, sequence_length = None):
        """Given a numpy array as input (and sequence_length,
        which is none for most kernels but must be specified
        for convolution kernels), generate random features
        as output. In addition, convert the input y-values
        to live on the same device that the kernel is currently
        on."""
        y_out = input_y
        if self.device == "cuda":
            y_out = cp.asarray(y_out)

        return self.transform_x(input_x, sequence_length), y_out


    def gradient_x(self, input_x, sequence_length = None):
        """Given a numpy array as input (and sequence_length,
        which is none for most kernels but must be specified
        for convolution kernels), generate random features
        and gradient as output."""
        # This always generates a copy, which means that we
        # are never working on the input data, only on a copy,
        # and can therefore modify it with impunity.
        if not input_x.flags["C_CONTIGUOUS"]:
            if self.double_precision:
                xin = np.ascontiguousarray(input_x, np.float64)
            else:
                xin = np.ascontiguousarray(input_x, np.float32)
        elif self.double_precision:
            xin = input_x.astype(np.float64, copy=True)
        else:
            xin = input_x.astype(np.float32, copy=True)

        if self.device == "cuda":
            xin = cp.asarray(xin)

        slen = None
        if sequence_length is not None:
            slen = sequence_length.astype(np.int32, copy=False)

        xtrans, xgrad = self.kernel_specific_gradient(xin, slen)
        if self.fit_intercept:
            xtrans[:,0] = 1.
            if xgrad.shape[2] > 0:
                xgrad[:,0,:] = 0.
        return xtrans, xgrad



    def gradient_x_y(self, input_x, input_y, sequence_length = None):
        """Given a numpy array as input (and sequence_length,
        which is none for most kernels but must be specified
        for convolution kernels), generate random features
        and gradient as output. In addition, convert the input y-values
        to live on the same device that the kernel is currently
        on."""
        y_out = input_y
        if self.device == "cuda":
            y_out = cp.asarray(y_out)

        xtrans, dz_dsigma = self.gradient_x(input_x, sequence_length)
        return xtrans, dz_dsigma, y_out




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

    def get_xdim(self):
        """Returns the _xdim value. No setter since this should
        not be set externally."""
        return self._xdim


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
            value (str): Must be one of 'cpu', 'cuda'.

        Raises:
            ValueError: A ValueError is raised if an unrecognized
                device is passed.
        """
        if value not in ('cpu', 'cuda'):
            raise ValueError("Unrecognized device supplied. Must be one "
                    "of 'cpu', 'cuda'.")
        self.device_ = value
        self.kernel_specific_set_device(value)
