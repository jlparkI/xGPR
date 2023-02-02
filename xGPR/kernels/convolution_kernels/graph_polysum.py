"""A polynomial kernel. Unlike the classic polynomial kernel,
this one is designed to be summed across pairwise comparisons
of all input vectors for two inputs, so it can be used for graphs
or time series that have different lengths."""
from math import ceil
import numpy as np
from scipy.stats import chi
try:
    import cupy as cp
    from cuda_convolution_double_hadamard_operations import doubleGpuGraphPolyFHT
    from cuda_convolution_float_hadamard_operations import floatGpuGraphPolyFHT
except:
    pass

from ..kernel_baseclass import KernelBaseclass
from cpu_convolution_double_hadamard_operations import doubleCpuGraphPolyFHT
from cpu_convolution_float_hadamard_operations import floatCpuGraphPolyFHT


class GraphPolySum(KernelBaseclass):
    """The GraphPolySum kernel corresponds to summing a polynomial
    kernel applied pairwise to each possible pairing of all
    input vectors associated with two datapoints (e.g. time series,
    graphs). Remarkably we can evaluate this in a very efficient way.

    Attributes:
        hyperparams (np.ndarray): This kernel has two
            hyperparameters: lambda_ (noise), beta_ (amplitude).
        polydegree (int): The degree of the polynomial to be applied.
        chi_arr (array): Array of shape (polydegree, init_calc_freqsize)
            for ensuring correct marginals on random feature generation.
        radem_diag (array): The diagonal Rademacher arrays for random
            feature generation.
        device (str): Either 'cpu' or 'gpu'; indicates where random
            features should be generated.
        graph_poly_func: A reference to the Cython-wrapped C function
            that will be used for random feature generation.
        padded_dims (int): The size of the expected input after zero
            padding.
        init_calc_freqsize (int): The number of features to generate.
            May be greater than num_rffs, in which case excess is
            discarded.
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
            raise ValueError("For the GraphPoly kernel, 'polydegree' must be "
                "included as the degree of the polynomial.")
        self.polydegree = kernel_spec_parms["polydegree"]
        if self.polydegree < 2 or self.polydegree > 4:
            raise ValueError("Polydegree should be in the range from 2 to 4.")
        self.hyperparams = np.ones((2))
        self.bounds = np.asarray([[1e-3,1e1], [0.2, 5]])

        self.padded_dims = 2**ceil(np.log2(max(xdim[2], 2)))
        num_repeats = ceil(self.num_freqs / self.padded_dims)
        self.init_calc_freqsize = num_repeats * self.padded_dims

        radem_array = np.asarray([-1,1], dtype=np.int8)
        rng = np.random.default_rng(random_seed)
        self.radem_diag = rng.choice(radem_array, size=(3 * self.polydegree,
                                1, self.init_calc_freqsize),
                                replace=True)

        self.chi_arr = chi.rvs(df=self.padded_dims,
                        size=(self.polydegree, self.init_calc_freqsize),
                            random_state = random_seed)

        self.graph_poly_func = None
        self.device = device
        self.chi_arr = self.chi_arr.astype(self.dtype)


    def kernel_specific_set_device(self, new_device):
        """Called when device is changed. Moves
        some of the object parameters to the appropriate device."""
        if new_device == "gpu":
            self.chi_arr = cp.asarray(self.chi_arr).astype(self.dtype)
            self.radem_diag = cp.asarray(self.radem_diag)
            if self.double_precision:
                self.graph_poly_func = doubleGpuGraphPolyFHT
            else:
                self.graph_poly_func = floatGpuGraphPolyFHT
        else:
            if not isinstance(self.radem_diag, np.ndarray):
                self.chi_arr = cp.asnumpy(self.chi_arr).astype(self.dtype)
                self.radem_diag = cp.asnumpy(self.radem_diag)
            if self.double_precision:
                self.graph_poly_func = doubleCpuGraphPolyFHT
            else:
                self.graph_poly_func = floatCpuGraphPolyFHT




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
        if len(input_x.shape) != 3:
            raise ValueError("Input to GraphPoly must be a 3d array.")
        if input_x.shape[2] != self.xdim[2]:
            raise ValueError("Unexpected number of features supplied to GraphPoly.")
        retyped_input = self.zero_arr((input_x.shape[0], input_x.shape[1],
                    self.padded_dims), self.dtype)
        retyped_input[:,:,1:input_x.shape[2] + 1] = input_x
        retyped_input[:,:,0] = 1
        output_x = self.zero_arr((input_x.shape[0], self.init_calc_freqsize),
                            dtype = self.dtype)
        self.graph_poly_func(retyped_input, self.radem_diag,
                self.chi_arr, output_x, self.polydegree, 2)
        output_x = output_x[:,:self.num_rffs].astype(self.out_type) * self.hyperparams[1]
        return output_x



    def kernel_specific_gradient(self, input_x):
        """Since all kernels share the beta and lambda hyperparameters,
        the gradient for these can be calculated by the parent class.
        This kernel has no kernel-specific hyperparameters and hence
        can return a shape[1] == 0 array for gradient.
        """
        xtrans = self.transform_x(input_x)
        return xtrans, np.zeros((xtrans.shape[0], 0, 0))
