"""A linear kernel, which corresponds to Bayesian linear
regression."""
import numpy as np
try:
    import cupy as cp
except:
    pass
from ..kernel_baseclass import KernelBaseclass


class Linear(KernelBaseclass):
    """The Linear kernel corresponds to Bayesian linear regression.
    For attributes not described here, see the baseclass.
    """

    def __init__(self, xdim, num_rffs, random_seed = 123,
                device = "cpu", num_threads = 2,
                double_precision = True,
                kernel_spec_parms = {}):
        """Constructor.

        Args:
            xdim (tuple): The dimensions of the input.
            num_rffs (int): The user-requested number of random Fourier features.
                For this kernel, it is ignored.
            random_seed (int): The seed to the random number generator.
            device (str): One of 'cpu', 'gpu'. Indicates the starting device.
            num_threads (int): The number of threads to use for random feature generation
                if running on CPU; if running on GPU this is ignored. Since random features
                are not generated for this kernel this is ignored.
            double_precision (bool): Not used for this kernel; accepted to preserve
                common interface with other kernels.
            kernel_spec_parms (dict): A dictionary of kernel-specific parameters.
                In this case, may optionally contain 'intercept'; if false, no y-intercept
                is fitted.
        """
        self.fit_intercept = True
        actual_rffs = xdim[1] + 1
        if "intercept" in kernel_spec_parms:
            if kernel_spec_parms["intercept"] is False:
                self.fit_intercept = False
                actual_rffs = xdim[1]

        super().__init__(actual_rffs, xdim, kernel_spec_parms = kernel_spec_parms)

        if len(xdim) > 2:
            raise ValueError("The Linear kernel is only applicable for "
                    "fixed vector input.")
        self.hyperparams = np.ones((1))
        self.bounds = np.asarray([[1e-3,1e1]])

        self.device = device


    def kernel_specific_set_device(self, new_device):
        """Provided for consistency with baseclass. This
        kernel -- unlike all others -- has no device-specific
        params."""
        return


    def kernel_specific_set_hyperparams(self):
        """Provided for consistency with baseclass. This
        kernel has no kernel-specific properties that must
        be reset after hyperparameters are changed."""
        return


    def kernel_specific_transform(self, input_x, sequence_length = None):
        """Most kernels generate random features. For the LinearKernel,
        we merely return the input, adding an additional column for
        an intercept if specified.
        The sequence length argument is accepted for consistency with
        baseclass but is not used by this kernel.
        """
        if self.fit_intercept:
            if self.device == "cuda":
                xtrans = cp.zeros((input_x.shape[0], input_x.shape[1] + 1), cp.float64)
            else:
                xtrans = np.zeros((input_x.shape[0], input_x.shape[1] + 1), np.float64)
            xtrans[:,1:] = input_x
            return xtrans
        return input_x


    def kernel_specific_gradient(self, input_x, sequence_length = None):
        """The gradient for kernel-specific hyperparameters is calculated
        using an array (dz_dsigma) specific to each kernel.
        This kernel has no kernel-specific hyperparameters and hence
        can return a shape[1] == 0 array for gradient.
        """
        xtrans = self.transform_x(input_x)
        if self.device == "cuda":
            return xtrans, cp.zeros((xtrans.shape[0], 0, 0))
        return xtrans, np.zeros((xtrans.shape[0], 0, 0))
