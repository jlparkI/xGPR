"""Describes the SORFKernelBaseclass, which is used by the main
fixed-vector kernels -- RBF, Matern.

All of these share some methods and attributes in common
which are stored here to avoid redundancy."""
from abc import ABC
from math import ceil

import numpy as np
from scipy.stats import chi

from xGPR.xgpr_cpu_rfgen_cpp_ext import cpuRBFFeatureGen, cpuRBFGrad
try:
    import cupy as cp
    from xGPR.xgpr_cuda_rfgen_cpp_ext import cudaRBFFeatureGen, cudaRBFGrad
except:
    pass

from ..kernel_baseclass import KernelBaseclass



class SORFKernelBaseclass(KernelBaseclass, ABC):
    """The baseclass for structured orthogonal random features (SORF)
    kernels that accept fixed-vector inputs. Since it inherits from
    KernelBaseclass, it includes the attributes of that class. Only
    additional attributes unique to this class are described here.

    Attributes:
        radem_diag: The diagonal matrices for the SORF transform. Type is int8.
        chi_arr: A diagonal array whose elements are drawn from the chi
            distribution. Ensures the marginals of the matrix resulting
            from S H D1 H D2 H D3 are correct.
    """

    def __init__(self, num_rffs, xdim, num_threads = 2,
                            sine_cosine_kernel = True, random_seed = 123,
                            double_precision = False,
                            kernel_spec_parms = {}):
        """Constructor for the SORFKernelBaseclass. Calls the KernelBaseclass
        constructor first.

        Args:
            num_rffs (int): The user-requested number of random Fourier features.
                For sine-cosine kernels (RBF, Matern), this will be saved by the
                class as num_rffs.
            xdim (tuple): The dimensions of the input. Either (N, D) or (N, M, D)
                where N is the number of datapoints, D is number of features
                and M is number of timepoints or sequence elements (convolution
                kernels only).
            num_threads (int): The number of threads to use for generating random
                features if running on CPU. If running on GPU, this is ignored.
            random_seed (int): The seed to the random number generator.
            sine_cosine_kernel (bool): If True, the kernel is a sine-cosine kernel,
                meaning it will sample self.num_freqs frequencies and use the sine
                and cosine of each to generate twice as many features
                (self.num_rffs). sine-cosine kernels only accept num_rffs
                that are even numbers.
            double_precision (bool): If True, generate random features in double precision.
                Otherwise, generate as single precision.
            kernel_spec_parms (dict): A dictionary of other kernel-specific settings.

        Raises:
            ValueError: If a non 2d input array dimensionality is supplied.
        """
        super().__init__(num_rffs, xdim, num_threads = num_threads,
                sine_cosine_kernel = sine_cosine_kernel,
                double_precision = double_precision,
                kernel_spec_parms = kernel_spec_parms)
        if len(xdim) != 2:
            raise ValueError("The dimensionality of the input is inappropriate for "
                        "the kernel you have selected.")

        padded_dims = 2**ceil(np.log2(max(xdim[-1], 2)))

        radem_array = np.asarray([-1,1], dtype=np.int8)
        rng = np.random.default_rng(random_seed)
        if padded_dims < self.num_freqs:
            nblocks = ceil(self.num_freqs / padded_dims)
        else:
            nblocks = 1
        self.radem_diag = rng.choice(radem_array, size=(3, 1,
                nblocks * padded_dims), replace=True)
        self.chi_arr = chi.rvs(df=padded_dims, size=self.num_freqs,
                            random_state = random_seed)
        if not self.double_precision:
            self.chi_arr = self.chi_arr.astype(np.float32)



    def kernel_specific_set_device(self, new_device):
        """Called by parent class when device is changed. Moves
        some of the object parameters to the appropriate device
        and updates self.cosfunc, self.sinfunc, which are
        convenience references to np.cos / np.sin or cp.cos
        / cp.sin."""
        if new_device == "cpu":
            if not isinstance(self.radem_diag, np.ndarray):
                self.radem_diag = cp.asnumpy(self.radem_diag)
                self.chi_arr = cp.asnumpy(self.chi_arr)
        else:
            self.radem_diag = cp.asarray(self.radem_diag)
            self.chi_arr = cp.asarray(self.chi_arr)



    def kernel_specific_transform(self, input_x, sequence_length = None):
        """Combines the two steps involved in random feature generation
        to generate random features.

        Args:
            xtrans: Either a cupy or numpy array containing the input.
            sequence_length: Accepted for consistency with baseclass and
                kernels that use this argument but is not used by this
                class of kernels and is therefore ignored.

        Returns:
            xtrans: A cupy or numpy array containing the generated features.
        """
        input_x *= self.hyperparams[1]
        if self.device == "cpu":
            output_x = np.zeros((input_x.shape[0], self.num_rffs), np.float64)
            cpuRBFFeatureGen(input_x, output_x, self.radem_diag, self.chi_arr,
                self.num_threads, self.fit_intercept)
        else:
            output_x = cp.zeros((input_x.shape[0], self.num_rffs), cp.float64)
            cudaRBFFeatureGen(input_x, output_x, self.radem_diag, self.chi_arr,
                self.fit_intercept)
        return output_x


    def kernel_specific_set_hyperparams(self):
        """Provided for consistency with baseclass. This
        kernel has no kernel-specific properties that must
        be reset after hyperparameters are changed."""
        return


    def kernel_specific_gradient(self, input_x, sequence_length = None):
        """The gradient for kernel-specific hyperparameters is calculated
        using an array (dz_dsigma) specific to each kernel.

        Args:
            input_x: A cupy or numpy array containing the raw input data.
            sequence_length: Accepted for consistency with baseclass and
                kernels that use this argument but is not used by this
                class of kernels and is therefore ignored.

        Returns:
            output_x: A cupy or numpy array containing the random feature
                representation of the input.
            dz_dsigma: A cupy or numpy array containing the derivative of
                output_x with respect to the kernel-specific hyperparameters.
        """
        if self.device == "cpu":
            output_x = np.zeros((input_x.shape[0], self.num_rffs), np.float64)
            dz_dsigma = np.zeros((input_x.shape[0], self.num_rffs, 1), np.float64)
            cpuRBFGrad(input_x, output_x, dz_dsigma, self.radem_diag, self.chi_arr,
                self.hyperparams[1], self.num_threads, self.fit_intercept)
        else:
            output_x = cp.zeros((input_x.shape[0], self.num_rffs), cp.float64)
            dz_dsigma = cp.zeros((input_x.shape[0], self.num_rffs, 1), cp.float64)
            cudaRBFGrad(input_x, output_x, dz_dsigma, self.radem_diag, self.chi_arr,
                self.hyperparams[1], self.fit_intercept)
        return output_x, dz_dsigma
