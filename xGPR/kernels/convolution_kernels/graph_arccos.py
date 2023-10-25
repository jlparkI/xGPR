"""This graph kernel applies an arc-cosine kernel of order
1 or 2 to each node and averages -- equivalent to assessing
all possible pairwise comparisons between two inputs using
an arc-cosine kernel (ReLU activation)."""
from math import ceil

import numpy as np
from scipy.stats import chi
try:
    import cupy as cp
    from cuda_rf_gen_module import gpuConv1dArcCosFGen
except:
    pass

from ..kernel_baseclass import KernelBaseclass
from cpu_rf_gen_module import cpuConv1dArcCosFGen


class GraphArcCosine(KernelBaseclass):
    """This class implements a GraphArcCosine kernel that averages over
    all possible pairwise comparisons between two graphs, as computed
    using an arc-cosine kernel of order 1 or 2. Unlike
    GraphRBF, it has no additional hyperparameters beyond the two
    shared by all kernels.
    This class inherits from KernelBaseclass.
    Only attributes unique to this child are described in this docstring.

    Attributes:
        hyperparams (np.ndarray): This kernel has three
            hyperparameters: lambda_ (noise), beta_ (amplitude)
            and sigma (inverse mismatch tolerance).
        padded_dims (int): The size of the expected input data
            after zero-padding to be a power of 2.
        init_calc_freqsize (int): The number of times the transform
            will need to be performed to generate the requested number
            of sampled frequencies.
        order (int): The order of the kernel (either 1 or 2).
        effective_dim (int): The effective dimensionality of the input after
            adding 1 (to add a y-intercept).
        radem_diag: The diagonal matrices for the SORF transform. Type is int8.
        chi_arr: A diagonal array whose elements are drawn from the chi
            distribution. Ensures the marginals of the matrix resulting
            from S H D1 H D2 H D3 are correct.
        conv_func: A reference to the random feature generation function
            appropriate for the current device.
        fit_intercept (bool): Determines whether to fit a y-intercept.
            Defaults to True.
        graph_average (bool): If True, divide the summed random features for the
            graph by the number of nodes. Defaults to False. Can be set to
            True by supplying "averaging":True under kernel_spec_parms.
    """

    def __init__(self, xdim, num_rffs, random_seed = 123, device = "cpu",
                    num_threads = 2, double_precision = False, kernel_spec_parms = {}):
        """Constructor.

        Args:
            xdim (tuple): The dimensions of the input. Either (N, D) or (N, M, D)
                where N is the number of datapoints, D is number of features
                and M is number of timepoints or sequence elements (convolution
                kernels only).
            num_rffs (int): The user-requested number of random Fourier features.
            random_seed (int): The seed to the random number generator.
            device (str): One of 'cpu', 'gpu'. Indicates the starting device.
            num_threads (int): The number of threads to use for random feature generation
                if running on CPU. Ignored if running on GPU.
            double_precision (bool): If True, generate random features in double precision.
                Otherwise, generate as single precision.
            kernel_spec_parms (dict): A dictionary of kernel-specific parameters.
                In this case, must contain "order", which indicates whether the
                kernel is order 1 or order 2.

        Raises:
            ValueError: A ValueError is raised if the dimensions of the input are
                inappropriate given the conv_width.
        """
        super().__init__(num_rffs, xdim, num_threads, sine_cosine_kernel = False,
                double_precision = double_precision, kernel_spec_parms = kernel_spec_parms)
        if len(xdim) != 3:
            raise ValueError("Tried to initialize the GraphArcCos kernel with a "
                    "2d x-array! x should be a 3d array for a graph kernel.")
        self.graph_average = False
        if "averaging" in kernel_spec_parms:
            if kernel_spec_parms["averaging"]:
                self.graph_average = True

        if "order" not in kernel_spec_parms:
            raise ValueError("For the GraphArcCosine kernel, 'order' must be "
                "included to indicate an arc-cosine kernel of order 1 or 2.")
        if kernel_spec_parms["order"] not in [1,2]:
            raise ValueError("For the GraphArcCosine kernel, 'order' must be "
                "included and must be either 1 or 2.")

        self.effective_dim = xdim[2] + 1
        self.order = kernel_spec_parms["order"]

        self.hyperparams = np.ones((2))
        self.bounds = np.asarray([[1e-3,1e1], [0.2, 5]])
        rng = np.random.default_rng(random_seed)

        self.padded_dims = 2**ceil(np.log2(max(self.effective_dim, 2)))
        self.init_calc_freqsize = ceil(self.num_freqs / self.padded_dims) * \
                        self.padded_dims

        radem_array = np.asarray([-1,1], dtype=np.int8)

        self.radem_diag = rng.choice(radem_array, size=(3, 1, self.init_calc_freqsize),
                                replace=True)
        self.chi_arr = chi.rvs(df=self.padded_dims, size=self.num_freqs,
                            random_state = random_seed)

        self.conv_func = None
        self.device = device


    def kernel_specific_set_device(self, new_device):
        """Called by parent class when device is changed. Moves
        some of the object parameters to the appropriate device
        and updates self.conv_func, self.stride_tricks and
        self.contiguous_array, which are convenience references
        to the numpy / cupy versions of functions required
        for generating features."""
        if new_device == "gpu":
            self.conv_func = gpuConv1dArcCosFGen
            self.radem_diag = cp.asarray(self.radem_diag)
            self.chi_arr = cp.asarray(self.chi_arr).astype(self.dtype)
        else:
            if not isinstance(self.radem_diag, np.ndarray):
                self.radem_diag = cp.asnumpy(self.radem_diag)
                self.chi_arr = cp.asnumpy(self.chi_arr)
            else:
                self.chi_arr = self.chi_arr.astype(self.dtype)
            self.conv_func = cpuConv1dArcCosFGen
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
        xtrans = self.zero_arr((input_x.shape[0], self.num_rffs), self.out_type)
        reshaped_x = self.zero_arr((input_x.shape[0], input_x.shape[1],
                                self.padded_dims), self.dtype)
        reshaped_x[:,:,:input_x.shape[2]] = input_x
        reshaped_x[:,:,input_x.shape[2]] = 1.0
        self.conv_func(reshaped_x, self.radem_diag, xtrans, self.chi_arr,
                self.num_threads, self.hyperparams[1], self.order,
                self.fit_intercept)
        if self.graph_average:
            xtrans /= input_x.shape[1]
        return xtrans



    def kernel_specific_gradient(self, input_x):
        """Since all kernels share the beta and lambda hyperparameters,
        the gradient for these can be calculated by the parent class.
        This kernel has no kernel-specific hyperparameters and hence
        can return a shape[1] == 0 array for gradient.
        """
        xtrans = self.transform_x(input_x)
        return xtrans, np.zeros((xtrans.shape[0], 0, 0))
