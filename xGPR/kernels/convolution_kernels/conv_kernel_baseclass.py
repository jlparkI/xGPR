"""Describes the Conv1dKernelBaseclass, which is shared
as a parent class by several convolution kernels for
graphs and sequences."""
from abc import ABC
from math import ceil

import numpy as np
from scipy.stats import chi

from xGPR.xgpr_cpu_rfgen_cpp_ext import cpuConv1dFGen, cpuConvGrad
try:
    import cupy as cp
    from xGPR.xgpr_cuda_rfgen_cpp_ext import cudaConv1dFGen, cudaConvGrad
except:
    pass

from ..kernel_baseclass import KernelBaseclass



class ConvKernelBaseclass(KernelBaseclass, ABC):
    """The baseclass for structured orthogonal random features (SORF)
    kernels that do 1d convolution. Since it inherits from
    KernelBaseclass, it includes the attributes of that class. Only
    additional attributes unique to this class are described here.

    Attributes:
        conv_width (int): The width of the convolution kernel.
            This hyperparameter can be set based on experimentation
            or domain knowledge.
        radem_diag: The diagonal matrices for the SORF transform. Type is int8.
        chi_arr: A diagonal array whose elements are drawn from the chi
            distribution. Ensures the marginals of the matrix resulting
            from S H D1 H D2 H D3 are correct.
        sequence_average (bool): If True, the features are averaged over the sequence
            when summing. Can be set to True by passing "averaging":True under
            kernel_spec_parms, otherwise defaults to False. This is useful if
            modeling properties of a sequence that are not size-extensive.
    """

    def __init__(self, xdim, num_rffs, random_seed = 123,
                    num_threads = 2, double_precision = False,
                    conv_width = 9, kernel_spec_parms = {}):
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
            device (str): One of 'cpu', 'cuda'. Indicates the starting device.
            num_threads (int): The number of threads to use for random feature generation
                if running on CPU. If running on GPU, this is ignored.
            double_precision (bool): If True, generate random features in double precision.
                Otherwise, generate as single precision.
            conv_width (int): The width of the convolution to perform.
            kernel_spec_parms (dict): A dictionary of additional kernel-specific
                attributes.

        Raises:
            RuntimeError: A RuntimeError is raised if the dimensions of the input are
                inappropriate given the conv_width.
        """
        super().__init__(num_rffs, xdim, num_threads = 2,
                sine_cosine_kernel = True, double_precision = double_precision,
                kernel_spec_parms = kernel_spec_parms)
        if len(xdim) != 3:
            raise RuntimeError("Tried to initialize a Conv1d kernel with a 2d x-"
                    "array! x should be a 3d array for Conv1d.")

        self.scaling_type = 0
        if "averaging" in kernel_spec_parms:
            if kernel_spec_parms["averaging"] == "none":
                self.scaling_type = 0
            elif kernel_spec_parms["averaging"] == "sqrt":
                self.scaling_type = 1
            elif kernel_spec_parms["averaging"] == "full":
                self.scaling_type = 2
            else:
                raise RuntimeError("Unrecognized value for 'averaging' supplied, "
                        "should be one of 'none', 'sqrt', 'full'.")

        rng = np.random.default_rng(random_seed)
        self.conv_width = conv_width

        dim2_no_padding = self.conv_width * xdim[2]
        padded_dims = 2**ceil(np.log2(max(dim2_no_padding, 2)))
        init_calc_freqsize = ceil(self.num_freqs / padded_dims) * \
                        padded_dims

        radem_array = np.asarray([-1,1], dtype=np.int8)
        self.radem_diag = rng.choice(radem_array, size=(3, 1, init_calc_freqsize),
                                replace=True)
        self.chi_arr = chi.rvs(df=padded_dims, size=self.num_freqs,
                            random_state = random_seed)
        if not self.double_precision:
            self.chi_arr = self.chi_arr.astype(np.float32)




    def kernel_specific_set_device(self, new_device):
        """Called by parent class when device is changed. Moves
        some of the object parameters to the appropriate device."""
        if new_device == "cuda":
            self.radem_diag = cp.asarray(self.radem_diag)
            self.chi_arr = cp.asarray(self.chi_arr)
        elif not isinstance(self.radem_diag, np.ndarray):
            self.radem_diag = cp.asnumpy(self.radem_diag)
            self.chi_arr = cp.asnumpy(self.chi_arr)



    def kernel_specific_transform(self, input_x, sequence_length):
        """Generates random features.

        Args:
            input_x: A numpy or cupy array containing the raw input data.

        Returns:
            xtrans: A cupy or numpy array containing the generated features.
            sequence_length: A numpy or cupy array containing the number of
                elements in each sequence -- so that zero padding can be masked.

        Raises:
            RuntimeError: A value error is raised if the dimensionality of the
                input does not meet validity criteria.
        """
        if sequence_length is None:
            raise RuntimeError("sequence_length is required for convolution kernels.")
        if input_x.shape[2] != self._xdim[2]:
            raise RuntimeError("Unexpected input shape supplied.")

        input_x *= self.hyperparams[1]

        if self.device == "cpu":
            xtrans = np.zeros((input_x.shape[0], self.num_rffs), np.float64)
            cpuConv1dFGen(input_x, xtrans, self.radem_diag, self.chi_arr,
                    sequence_length, self.conv_width, self.scaling_type,
                    self.num_threads)
        else:
            xtrans = cp.zeros((input_x.shape[0], self.num_rffs), cp.float64)
            cudaConv1dFGen(input_x, xtrans, self.radem_diag, self.chi_arr,
                    sequence_length, self.conv_width, self.scaling_type)

        return xtrans


    def kernel_specific_set_hyperparams(self):
        """Provided for consistency with baseclass. This
        kernel has no kernel-specific properties that must
        be reset after hyperparameters are changed."""
        return


    def kernel_specific_gradient(self, input_x, sequence_length):
        """The gradient for kernel-specific hyperparameters is calculated
        using an array (dz_dsigma) specific to each kernel.

        Args:
            input_x: A cupy or numpy array containing the raw input data.
            sequence_length: A numpy or cupy array containing the number of
                elements in each sequence -- so that zero padding can be masked.

        Returns:
            output_x: A cupy or numpy array containing the random feature
                representation of the input.
            dz_dsigma: A cupy or numpy array containing the derivative of
                output_x with respect to the kernel-specific hyperparameters.
        """
        if sequence_length is None:
            raise RuntimeError("sequence_length is required for convolution kernels.")
        if input_x.shape[2] != self._xdim[2]:
            raise RuntimeError("Unexpected input shape supplied.")

        if self.device == "cpu":
            xtrans = np.zeros((input_x.shape[0], self.num_rffs), np.float64)
            dz_dsigma = np.zeros((input_x.shape[0], self.num_rffs, 1), np.float64)
            cpuConvGrad(input_x, xtrans, self.radem_diag, self.chi_arr,
                    sequence_length, dz_dsigma, self.hyperparams[1],
                    self.conv_width, self.scaling_type,
                    self.num_threads)
        else:
            xtrans = cp.zeros((input_x.shape[0], self.num_rffs), cp.float64)
            dz_dsigma = cp.zeros((input_x.shape[0], self.num_rffs, 1), cp.float64)
            cudaConvGrad(input_x, xtrans, self.radem_diag, self.chi_arr,
                    sequence_length, dz_dsigma, self.hyperparams[1],
                    self.conv_width, self.scaling_type)

        return xtrans, dz_dsigma
