'''The Matern kernel can handle "squiggly" or bumpy functions unlike the RBF,
but like the RBF is prone to "ringing" when discontinuities are encountered.
The closed form kernel is the Fourier transform of a student t distribution.'''
import numpy as np
from .sorf_kernel_baseclass import SORFKernelBaseclass



class Matern(SORFKernelBaseclass):
    """The Matern kernel often offers performance similar to RBF but
    sometimes with slight improvement owing to its greater
    ability to handle 'bumpy' functions. This class inherits from
    SORFKernelBaseclass which in turn inherits from KernelBaseclass.
    Only attributes unique to this child are described in this docstring.

    Attributes:
        hyperparams (np.ndarray): This kernel has three
            hyperparameters: lambda_ (noise), beta_ (amplitude)
            and sigma (inverse mismatch tolerance).
        matern_nu (float): The nu hyperparamter of the Matern kernel.
            Set by the user. Must be >= 1/2, <= 5/2. Larger values
            indicate smoother functions are expected. 5/2 is
            recommended.
    """

    def __init__(self, xdim, num_rffs, random_seed = 123, device = "cpu",
                double_precision = True, kernel_spec_parms = {}):
        """Constructor for Matern.

        Args:
            xdim (tuple): The dimensions of the input.
            num_rffs (int): The user-requested number of random Fourier features.
            random_seed (int): The seed to the random number generator.
            device (str): One of 'cpu', 'gpu'. Indicates the starting device.
            double_precision (bool): If True, generate random features in double precision.
                Otherwise, generate as single precision.
            kernel_spec_parms (dict): A dictionary of additional kernel-specific
                attributes. In this case, should contain 'matern_nu'.
        """
        super().__init__(num_rffs, xdim, double_precision,
                sine_cosine_kernel = True, random_seed = random_seed)
        self.hyperparams = np.ones((3))
        if "matern_nu" not in kernel_spec_parms:
            raise ValueError("Tried to initialize a Matern kernel without supplying nu.")

        self.matern_nu = kernel_spec_parms["matern_nu"]
        if self.matern_nu < 1/2 or self.matern_nu > 5/2:
            raise ValueError("nu must be >= 1/2 and <= 5/2.")
        rng = np.random.default_rng(random_seed)
        chisamples = np.sqrt(rng.chisquare(2 * self.matern_nu,
                                    size=self.num_freqs)
                                    / (self.matern_nu * 2) )
        self.chi_arr /= chisamples
        self.bounds = np.asarray([[1e-3,1e1], [0.125, 8], [1e-6, 1e2]])

        self.device = device
