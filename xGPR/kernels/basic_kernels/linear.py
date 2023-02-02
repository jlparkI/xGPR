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
    For other attributes not described here, see the baseclass.

    Attributes:
        fit_intercept (bool): Indicates whether we are fitting a
            y-intercept or no y-intercept.
        hyperparams (np.ndarray): This kernel has two
            hyperparameters: lambda_ (noise), beta_ (amplitude).
    """

    def __init__(self, xdim, num_rffs, random_seed = 123,
                device = "cpu", double_precision = True,
                kernel_spec_parms = {}):
        """Constructor.

        Args:
            xdim (tuple): The dimensions of the input.
            num_rffs (int): The user-requested number of random Fourier features.
                For this kernel, it is ignored.
            random_seed (int): The seed to the random number generator.
            device (str): One of 'cpu', 'gpu'. Indicates the starting device.
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
        super().__init__(actual_rffs, xdim)
        if len(xdim) > 2:
            raise ValueError("The Linear kernel is only applicable for "
                    "fixed vector input.")
        self.hyperparams = np.ones((2))
        self.bounds = np.asarray([[1e-3,1e1], [0.125, 8]])

        self.device = device


    def kernel_specific_set_device(self, new_device):
        """Provided for consistency with baseclass. This
        kernel -- unlike all others -- has no device-specific
        params."""
        return



    def transform_x(self, input_x):
        """Most kernels generate random features. For the LinearKernel,
        we merely return the input, adding an additional column for
        an intercept if specified.
        """
        if self.fit_intercept:
            xtrans = self.empty((input_x.shape[0], input_x.shape[1] + 1), self.out_type)
            xtrans[:,1:] = input_x
            xtrans[:,0] = 1
        else:
            xtrans = input_x.astype(self.out_type)
        return xtrans * self.hyperparams[1]


    def kernel_specific_gradient(self, input_x):
        """Since all kernels share the beta and lambda hyperparameters,
        the gradient for these can be calculated by the parent class.
        This kernel has no kernel-specific hyperparameters and hence
        can return a shape[1] == 0 array for gradient.
        """
        xtrans = self.transform_x(input_x)
        return xtrans, np.zeros((xtrans.shape[0], 0, 0))
