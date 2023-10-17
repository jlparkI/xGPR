"""Implements exact polynomial regression for a quadratic.
Note that this will become impractical if the number of
features in the input vector is large -- even 600 is on
the large side."""
import numpy as np
try:
    from cuda_rf_gen_module import cudaExactQuadratic
except:
    pass

from ..kernel_baseclass import KernelBaseclass
from cpu_rf_gen_module import cpuExactQuadratic


class ExactQuadratic(KernelBaseclass):
    """An exact quadratic, not approximated, implemented as
    polynomial regression.

    Attributes:
        hyperparams (np.ndarray): This kernel has two
            hyperparameters: lambda_ (noise), beta_ (amplitude).
        poly_func: A reference to the Cython-wrapped C function
            that will be used for feature generation.
    """

    def __init__(self, xdim, num_rffs, random_seed = 123,
                device = "cpu", num_threads = 2, double_precision = False,
                kernel_spec_parms = {}):
        """Constructor.

        Args:
            xdim (tuple): The dimensions of the input. Only 3d arrays are
                accepted, where shape[1] is the number of vertices in a graph
                and shape[2] is the number of features per vertex. For a fixed
                vector input, shape[1] can be 1.
            num_rffs (int): The user-requested number of random Fourier features.
                For this kernel only, this argument is ignored. It is preserved
                for consistency with other kernels.
            random_seed (int): The seed to the random number generator.
            device (str): One of 'cpu', 'gpu'. Indicates the starting device.
            num_threads (int): The number of threads to use if running on CPU. If
                running on GPU, this is ignored.
            double_precision (bool): If True, generate random features in double precision.
                Otherwise, generate as single precision.
            kernel_spec_parms (dict): A dictionary of kernel-specific parameters.
                In this case, may optionally contain 'intercept'; if false, no
                y-intercept is fitted.
        """
        actual_num_rffs = 1 + xdim[1] * 2 + int((xdim[1] * (xdim[1] - 1)) / 2)

        super().__init__(actual_num_rffs, xdim, num_threads,
                sine_cosine_kernel = False, double_precision = double_precision,
                kernel_spec_parms = kernel_spec_parms)

        self.hyperparams = np.ones((2))
        self.bounds = np.asarray([[1e-3,1e1], [0.1, 10]])

        self.poly_func = None
        self.device = device



    def kernel_specific_set_device(self, new_device):
        """Called when device is changed. Moves
        some of the object parameters to the appropriate device."""
        if new_device == "gpu":
            self.poly_func = cudaExactQuadratic
        else:
            self.poly_func = cpuExactQuadratic



    def transform_x(self, input_x):
        """Generates random features.

        Args:
            input_x: A cupy or numpy array depending on self.device
                containing the input data.

        Returns:
            output_x: A cupy or numpy array depending on self.device
                containing the results of random feature generation. Note
                that num_freqs rffs are generated, not num_rffs.
        """
        if len(input_x.shape) != 2:
            raise ValueError("Input to ClassicPoly must be a 2d array.")
        retyped_input = input_x.astype(self.dtype)
        output_x = self.zero_arr((input_x.shape[0], self.num_rffs),
                self.out_type)
        output_x[:,-1] = 1

        self.poly_func(retyped_input, output_x, self.num_threads)
        output_x *= self.hyperparams[1]
        return output_x


    def kernel_specific_set_hyperparams(self):
        """Provided for consistency with baseclass. This
        kernel has no kernel-specific properties that must
        be reset after hyperparameters are changed."""
        return


    def kernel_specific_gradient(self, input_x):
        """Since all kernels share the beta and lambda hyperparameters,
        the gradient for these can be calculated by the parent class.
        This kernel has no kernel-specific hyperparameters and hence
        can return a shape[1] == 0 array for gradient.
        """
        xtrans = self.transform_x(input_x)
        return xtrans, np.zeros((xtrans.shape[0], 0, 0))
