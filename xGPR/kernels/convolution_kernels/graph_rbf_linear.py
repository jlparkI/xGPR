"""This kernel is the sum of a GraphRBF kernel -- which applies a pairwise
RBF kernel across all nodes in two graphs -- and a Linear kernel, which
applies a linear function to the sum of all nodes in a graph."""
from math import ceil

import numpy as np
from scipy.stats import chi
try:
    import cupy as cp
    from cuda_rf_gen_module import doubleGpuConv1dFGen, doubleGpuConvGrad
    from cuda_rf_gen_module import floatGpuConv1dFGen, floatGpuConvGrad
except:
    pass

from ..kernel_baseclass import KernelBaseclass
from cpu_rf_gen_module import doubleCpuConv1dFGen, doubleCpuConvGrad
from cpu_rf_gen_module import floatCpuConv1dFGen, floatCpuConvGrad


class GraphRBFLinear(KernelBaseclass):
    """This kernel is the sum of a Linear kernel and a GraphRBF kernel.
    The GraphRBF kernel measures similarity using a pairwise RBF kernel
    across all nodes in two graphs, while the Linear kernel is a linear
    kernel across the sum of the node-features in the two graphs.
    This class inherits from KernelBaseclass. Only attributes unique to
    this child are described in this docstring.

    Attributes:
        hyperparams (np.ndarray): This kernel has three
            hyperparameters: lambda_ (noise), beta_ (amplitude)
            and sigma (inverse mismatch tolerance).
        padded_dims (int): The size of the expected input data
            after zero-padding to be a power of 2.
        internal_rffs (int): The number of random features that will be generated.
            This is different than num_rffs (from the parent class), which is what
            the kernel will report to anyone asking how many features it generates.
            The reason for this difference is that the linear + rbf kernel
            concatenates the input (for the linear portion of the kernel) to
            the random features generated for the RBF portion.
        init_calc_freqsize (int): The number of times the transform
            will need to be performed to generate the requested number
            of sampled frequencies.
        init_calc_featsize (int): The number of features generated initially
            (before discarding excess).
        radem_diag: The diagonal matrices for the SORF transform. Type is int8.
        chi_arr: A diagonal array whose elements are drawn from the chi
            distribution. Ensures the marginals of the matrix resulting
            from S H D1 H D2 H D3 are correct.
        conv_func: A reference to the random feature generation function
            appropriate for the current device.
        grad_func: A reference to the random feature generation & gradient
            calculation function appropriate for the current device.
    """

    def __init__(self, xdim, num_rffs, random_seed = 123, device = "cpu",
                    num_threads = 2, double_precision = False, **kwargs):
        """Constructor.

        Args:
            xdim (tuple): The dimensions of the input. Either (N, D) or (N, M, D)
                where N is the number of datapoints, D is number of features
                and M is number of timepoints or sequence elements (convolution
                kernels only).
            num_rffs (int): The user-requested number of random Fourier features.
                For sine-cosine kernels (RBF, Matern), this will be saved by the
                class as num_rffs.
            random_seed (int): The seed to the random number generator.
            device (str): One of 'cpu', 'gpu'. Indicates the starting device.
            num_threads (int): The number of threads to use for random feature generation
                if running on CPU. Ignored if running on GPU.
            double_precision (bool): If True, generate random features in double precision.
                Otherwise, generate as single precision.

        Raises:
            ValueError: A ValueError is raised if the dimensions of the input are
                inappropriate given the conv_width.
        """
        #Although this IS a sine_cosine kernel, we don't want the parent class
        #to enforce that num_rffs be a multiple of two (only needs to be true
        #for internal rffs), so we set sine_cosine_kernel to False.
        super().__init__(num_rffs, xdim, num_threads, sine_cosine_kernel = False,
                double_precision = double_precision)
        if len(xdim) != 3:
            raise ValueError("Tried to initialize the GraphRBFLinear kernel with a "
                    "2d x-array! x should be a 3d array for GraphRBFLinear.")
        self.internal_rffs = num_rffs - xdim[2]
        if self.internal_rffs <= 1 or not (self.internal_rffs / 2).is_integer():
            raise ValueError("For the GraphRBFLinear kernel, the number of 'random' "
                    "features requested includes the number of features in "
                    "the input. So, for example, if the input is a 90 x 100 "
                    "matrix for each graph and training_rffs is 1000, 900 random features will "
                    "be generated and the input features will be concatenated to "
                    "this to yield 1000 'random' features. The number of "
                    "training and fitting rffs requested should therefore be "
                    "at least num_node_features + 2, and after the input length "
                    "is subtracted, the remainder should be a power of two. The number of "
                    "variance_rffs requested is not affected.")

        self.hyperparams = np.ones((4))
        self.bounds = np.asarray([[1e-3,1e1], [0.1, 10], [1e-2, 15], [1e-2, 1e2]])
        rng = np.random.default_rng(random_seed)

        #Need to override parent class calculation of num_freqs to use
        #internal rffs.
        self.num_freqs = int(self.internal_rffs / 2)

        self.padded_dims = 2**ceil(np.log2(max(xdim[2], 2)))
        self.init_calc_freqsize = ceil(self.num_freqs / self.padded_dims) * \
                        self.padded_dims
        self.init_calc_featsize = 2 * self.init_calc_freqsize

        radem_array = np.asarray([-1,1], dtype=np.int8)

        self.radem_diag = rng.choice(radem_array, size=(3, 1, self.init_calc_freqsize),
                                replace=True)
        self.chi_arr = chi.rvs(df=self.padded_dims, size=self.num_freqs,
                            random_state = random_seed)

        self.conv_func = None
        self.grad_func = None
        self.device = device


    def kernel_specific_set_device(self, new_device):
        """Called by parent class when device is changed. Moves
        some of the object parameters to the appropriate device
        and updates self.conv_func, self.stride_tricks and
        self.contiguous_array, which are convenience references
        to the numpy / cupy versions of functions required
        for generating features."""
        if new_device == "gpu":
            if self.double_precision:
                self.conv_func = doubleGpuConv1dFGen
                self.grad_func = doubleGpuConvGrad
            else:
                self.conv_func = floatGpuConv1dFGen
                self.grad_func = floatGpuConvGrad
            self.radem_diag = cp.asarray(self.radem_diag)
            self.chi_arr = cp.asarray(self.chi_arr).astype(self.dtype)
        else:
            if not isinstance(self.radem_diag, np.ndarray):
                self.radem_diag = cp.asnumpy(self.radem_diag)
                self.chi_arr = cp.asnumpy(self.chi_arr)
            else:
                self.chi_arr = self.chi_arr.astype(self.dtype)
            if self.double_precision:
                self.conv_func = doubleCpuConv1dFGen
                self.grad_func = doubleCpuConvGrad
            else:
                self.conv_func = floatCpuConv1dFGen
                self.grad_func = floatCpuConvGrad
            self.chi_arr = self.chi_arr.astype(self.dtype)


    def kernel_specific_set_hyperparams(self):
        """Provided for consistency with baseclass. This
        kernel has no kernel-specific properties that must
        be reset after hyperparameters are changed."""
        return


    def transform_x(self, input_x):
        """Generates random features.

        Args:
            input_x: A numpy or cupy array containing the raw input data.

        Returns:
            xtrans: A cupy or numpy array containing the generated features.

        Raises:
            ValueError: A value error is raised if the dimensionality of the
                input does not meet validity criteria.
        """
        if len(input_x.shape) != 3:
            raise ValueError("Input X must be a 3d array.")
        xtrans = self.zero_arr((input_x.shape[0], self.internal_rffs), self.out_type)
        reshaped_x = self.zero_arr((input_x.shape[0], input_x.shape[1],
                                self.padded_dims), self.dtype)
        reshaped_x[:,:,:input_x.shape[2]] = input_x * self.hyperparams[3]
        output_x = self.empty((input_x.shape[0], self.num_rffs), self.out_type)

        self.conv_func(reshaped_x, self.radem_diag, xtrans, self.chi_arr,
                self.num_threads, self.hyperparams[1])
        output_x[:,:self.internal_rffs] = xtrans
        output_x[:,self.internal_rffs:] = cp.sum(input_x, axis=1) * self.hyperparams[1] * self.hyperparams[2]
        return output_x



    def kernel_specific_gradient(self, input_x):
        """Since all kernels share the beta and lambda hyperparameters,
        the gradient for these can be calculated by the parent class.
        The gradient kernel-specific hyperparameters however is calculated
        using an array (dz_dsigma) specific to each
        kernel. The kernel-specific arrays are calculated here.

        Args:
            input_x: A cupy or numpy array containing the raw input data.

        Returns:
            output_x: A cupy or numpy array containing the random feature
                representation of the input.
            dz_dsigma: A cupy or numpy array containing the derivative of
                output_x with respect to the kernel-specific hyperparameters.
        """
        if len(input_x.shape) != 3:
            raise ValueError("Input X must be a 3d array.")
        random_features = self.zero_arr((input_x.shape[0], self.internal_rffs),
                self.out_type)
        output_x = self.empty((input_x.shape[0], self.num_rffs), self.out_type)
        output_grad = self.zero_arr((input_x.shape[0], self.num_rffs, 2), self.out_type)
        reshaped_x = self.zero_arr((input_x.shape[0], input_x.shape[1],
                                self.padded_dims), self.dtype)
        reshaped_x[:,:,:input_x.shape[2]] = input_x
        output_grad[:,:self.internal_rffs,1:2] = self.grad_func(reshaped_x, self.radem_diag,
                random_features, self.chi_arr, self.num_threads, self.hyperparams[2],
                self.hyperparams[1])

        output_grad[:,self.internal_rffs:,0] = cp.sum(input_x, axis=1) * self.hyperparams[2]
        output_x[:,:self.internal_rffs] = random_features
        output_x[:,self.internal_rffs:] = output_grad[:,self.internal_rffs:,0] * self.hyperparams[1]
        return output_x, output_grad
