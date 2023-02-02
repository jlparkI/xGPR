"""The 'classic' polynomial kernel, implemented using
random features. Only degrees up to 4 are allowed; for much
larger degrees, it makes more sense to use RBF in general."""
from math import ceil
import numpy as np
from scipy.stats import chi
try:
    import cupy as cp
    from cuda_convolution_double_hadamard_operations import doubleGpuPolyFHT
    from cuda_convolution_float_hadamard_operations import floatGpuPolyFHT
except:
    pass

from ..kernel_baseclass import KernelBaseclass
from cpu_convolution_double_hadamard_operations import doubleCpuPolyFHT
from cpu_convolution_float_hadamard_operations import floatCpuPolyFHT


class Polynomial(KernelBaseclass):
    """The classic polynomial kernel, implemented using RF.
    Only attributes unique to this kernel are described in this
    docstring, see also parent class.

    Attributes:
        hyperparams (np.ndarray): This kernel has two
            hyperparameters: lambda_ (noise), beta_ (amplitude).
        polydegree (int): The degree of the polynomial to be applied.
        chi_arr (array): Array of shape (polydegree, init_calc_freqsize)
            for ensuring correct marginals on random feature generation.
        radem_diag (array): The diagonal Rademacher arrays for random
            feature generation.
        poly_func: A reference to the Cython-wrapped C function
            that will be used for random feature generation.
        padded_dims (int): The size of the expected input after zero
            padding.
        nblocks (int): The poly kernel transform is performed in blocks
            that result from H D1, where H is the
            normalized Hadamard matrix and D1 is a diagonal
            matrices whose elements are drawn from a Rademacher
            distribution. nblocks is the number of such operations
            required given the input size and number of random
            features requested.
    """

    def __init__(self, xdim, num_rffs, random_seed = 123,
                device = "cpu", double_precision = True,
                kernel_spec_parms = {}):
        """Constructor.

        Args:
            xdim (tuple): The dimensions of the input. Only 3d arrays are
                accepted, where shape[1] is the number of vertices in a graph
                and shape[2] is the number of features per vertex. For a fixed
                vector input, shape[1] can be 1.
            num_rffs (int): The user-requested number of random Fourier features.
            random_seed (int): The seed to the random number generator.
            device (str): One of 'cpu', 'gpu'. Indicates the starting device.
            double_precision (bool): If True, generate random features in double precision.
                Otherwise, generate as single precision.
            kernel_spec_parms (dict): A dictionary of kernel-specific parameters.
                In this case, must contain "polydegree", which is the degree
                of the polynomial.
        """

        super().__init__(num_rffs, xdim, double_precision, sine_cosine_kernel = False)
        if "polydegree" not in kernel_spec_parms:
            raise ValueError("For the Poly kernel, 'polydegree' must be "
                "included as the degree of the polynomial.")
        self.polydegree = kernel_spec_parms["polydegree"]
        if self.polydegree < 2 or self.polydegree > 4:
            raise ValueError("Polydegree should be in the range from 2 to 4.")

        self.hyperparams = np.ones((2))
        self.bounds = np.asarray([[1e-3,1e1], [0.1, 10]])
        self.padded_dims = 2**ceil(np.log2(max(self.xdim[-1] + 1, 2)))

        radem_array = np.asarray([-1,1], dtype=np.int8)
        rng = np.random.default_rng(random_seed)
        if self.padded_dims < self.num_freqs:
            self.nblocks = ceil(self.num_freqs / self.padded_dims)
        else:
            self.nblocks = 1
        self.radem_diag = rng.choice(radem_array, size=(3 * self.polydegree,
                        self.nblocks, self.padded_dims), replace=True)
        self.chi_arr = chi.rvs(df=self.padded_dims,
                size=(self.polydegree, self.nblocks, self.padded_dims),
                            random_state = random_seed)

        self.poly_func = None
        self.device = device
        self.chi_arr = self.chi_arr.astype(self.dtype)



    def kernel_specific_set_device(self, new_device):
        """Called when device is changed. Moves
        some of the object parameters to the appropriate device."""
        if new_device == "gpu":
            self.radem_diag = cp.asarray(self.radem_diag)
            self.chi_arr = cp.asarray(self.chi_arr)
            if self.double_precision:
                self.poly_func = doubleGpuPolyFHT
            else:
                self.poly_func = floatGpuPolyFHT
        else:
            if not isinstance(self.radem_diag, np.ndarray):
                self.radem_diag = cp.asnumpy(self.radem_diag)
                self.chi_arr = cp.asnumpy(self.chi_arr)
            if self.double_precision:
                self.poly_func = doubleCpuPolyFHT
            else:
                self.poly_func = floatCpuPolyFHT



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
        retyped_input = self.zero_arr((input_x.shape[0], self.nblocks,
                    self.padded_dims), self.dtype)
        retyped_input[:,:,1:input_x.shape[1] + 1] = input_x[:,None,:]
        retyped_input[:,:,0] = 1
        output_x = self.zero_arr((input_x.shape[0], self.nblocks,
                        self.padded_dims), dtype = self.dtype)

        self.poly_func(retyped_input, self.radem_diag,
                self.chi_arr, output_x, self.polydegree, 2)
        output_x = output_x.reshape((output_x.shape[0], output_x.shape[1] *
                        output_x.shape[2]))[:,:self.num_rffs].astype(self.out_type)
        output_x *= self.hyperparams[1]
        return output_x



    def kernel_specific_gradient(self, input_x):
        """Since all kernels share the beta and lambda hyperparameters,
        the gradient for these can be calculated by the parent class.
        This kernel has no kernel-specific hyperparameters and hence
        can return a shape[1] == 0 array for gradient.
        """
        xtrans = self.transform_x(input_x)
        return xtrans, np.zeros((xtrans.shape[0], 0, 0))
