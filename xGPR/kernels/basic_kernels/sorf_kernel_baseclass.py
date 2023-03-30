"""Describes the SORFKernelBaseclass, which is used by the main
fixed-vector kernels -- RBF, Matern.

All of these share some methods and attributes in common
which are stored here to avoid redundancy."""
from abc import ABC
from math import ceil

import numpy as np
from scipy.stats import chi
from cpu_rf_gen_module import doubleCpuRBFFeatureGen as dRBF
from cpu_rf_gen_module import floatCpuRBFFeatureGen as fRBF
from cpu_rf_gen_module import doubleCpuRBFGrad as dRBFGrad
from cpu_rf_gen_module import floatCpuRBFGrad as fRBFGrad

try:
    import cupy as cp
    from cuda_rf_gen_module import doubleCudaRBFFeatureGen as dCudaRBF
    from cuda_rf_gen_module import floatCudaRBFFeatureGen as fCudaRBF
    from cuda_rf_gen_module import doubleCudaRBFGrad as dCudaRBFGrad
    from cuda_rf_gen_module import floatCudaRBFGrad as fCudaRBFGrad
except:
    pass
from ..kernel_baseclass import KernelBaseclass

class SORFKernelBaseclass(KernelBaseclass, ABC):
    """The baseclass for structured orthogonal random features (SORF)
    kernels that accept fixed-vector inputs. Since it inherits from
    KernelBaseclass, it includes the attributes of that class. Only
    additional attributes unique to this class are described here.

    Attributes:
        nblocks (int): The SORF transform is performed in blocks
            that result from H D1 H D2 H D3, where H is the
            normalized Hadamard matrix and D1, D2, D3 are diagonal
            matrices whose elements are drawn from a Rademacher
            distribution. nblocks is the number of such operations
            required given the input size and number of random
            features requested.
        padded_dims (int): The next largest power of two greater than
            xdim[-1], since the Hadamard transform only operates on
            vectors whose length is a power of two.
        radem_diag: The diagonal matrices for the SORF transform. Type is int8.
        chi_arr: A diagonal array whose elements are drawn from the chi
            distribution. Ensures the marginals of the matrix resulting
            from S H D1 H D2 H D3 are correct.
        feature_gen: A reference to the wrapped C function that generates
            features.
        gradfun: A reference to the wrapped C function that calculates
            gradients.
    """

    def __init__(self, num_rffs, xdim, num_threads = 2,
                            sine_cosine_kernel = True, random_seed = 123,
                            double_precision = False):
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

        Raises:
            ValueError: If a non 2d input array dimensionality is supplied.
        """
        super().__init__(num_rffs, xdim, num_threads = num_threads,
                sine_cosine_kernel = sine_cosine_kernel,
                double_precision = double_precision)
        if len(xdim) != 2:
            raise ValueError("The dimensionality of the input is inappropriate for "
                        "the kernel you have selected.")

        self.padded_dims = 2**ceil(np.log2(max(xdim[-1], 2)))

        radem_array = np.asarray([-1,1], dtype=np.int8)
        rng = np.random.default_rng(random_seed)
        if self.padded_dims < self.num_freqs:
            self.nblocks = ceil(self.num_freqs / self.padded_dims)
        else:
            self.nblocks = 1
        self.radem_diag = rng.choice(radem_array, size=(3, self.nblocks, self.padded_dims),
                                replace=True)
        self.chi_arr = chi.rvs(df=self.padded_dims, size=self.num_freqs,
                            random_state = random_seed)

        self.feature_gen = fRBF
        self.gradfun = fRBFGrad



    def kernel_specific_set_device(self, new_device):
        """Called by parent class when device is changed. Moves
        some of the object parameters to the appropriate device
        and updates self.cosfunc, self.sinfunc, which are
        convenience references to np.cos / np.sin or cp.cos
        / cp.sin."""
        if new_device == "cpu":
            if self.double_precision:
                self.gradfun = dRBFGrad
                self.feature_gen = dRBF
            else:
                self.gradfun = fRBFGrad
                self.feature_gen = fRBF
            if not isinstance(self.radem_diag, np.ndarray):
                self.radem_diag = cp.asnumpy(self.radem_diag)
                self.chi_arr = cp.asnumpy(self.chi_arr)
            self.chi_arr = self.chi_arr.astype(self.dtype)
        else:
            if self.double_precision:
                self.gradfun = dCudaRBFGrad
                self.feature_gen = dCudaRBF
            else:
                self.gradfun = fCudaRBFGrad
                self.feature_gen = fCudaRBF
            self.radem_diag = cp.asarray(self.radem_diag)
            self.chi_arr = cp.asarray(self.chi_arr).astype(self.dtype)



    def transform_x(self, input_x):
        """Combines the two steps involved in random feature generation
        to generate random features.

        Args:
            input_x: Either a cupy or numpy array containing the input.

        Returns:
            xtrans: A cupy or numpy array containing the generated features.
        """
        xtrans = self.zero_arr((input_x.shape[0], self.nblocks, self.padded_dims),
                            dtype = self.dtype)
        xtrans[:,:,:self.xdim[1]] = input_x[:,None,:] * self.hyperparams[2]
        output_x = self.empty((input_x.shape[0], self.num_rffs), self.out_type)
        self.feature_gen(xtrans, output_x, self.radem_diag, self.chi_arr,
                self.hyperparams[1], self.num_threads)
        return output_x


    def kernel_specific_set_hyperparams(self):
        """Provided for consistency with baseclass. This
        kernel has no kernel-specific properties that must
        be reset after hyperparameters are changed."""
        return


    def kernel_specific_gradient(self, input_x):
        """Since all kernels share the beta and lambda hyperparameters,
        the gradient for these can be calculated by the parent class.
        The gradient kernel-specific hyperparameters however is calculated
        using an array (dz_dsigma) specific to each
        kernel. The kernel-specific arrays are calculated here.

        Args:
            input_x: A cupy or numpy array containing the raw input data.
            multiply_by_beta (bool): If False, skip multiplying by the amplitude
                hyperparameter. Useful for certain hyperparameter tuning
                routines.

        Returns:
            output_x: A cupy or numpy array containing the random feature
                representation of the input.
            dz_dsigma: A cupy or numpy array containing the derivative of
                output_x with respect to the kernel-specific hyperparameters.
        """
        xtrans = self.zero_arr((input_x.shape[0], self.nblocks, self.padded_dims),
                            dtype = self.dtype)
        xtrans[:,:,:self.xdim[1]] = input_x[:,None,:]
        output_x = self.empty((input_x.shape[0], self.num_rffs), self.out_type)
        dz_dsigma = self.gradfun(xtrans, output_x, self.radem_diag, self.chi_arr,
                self.hyperparams[1], self.hyperparams[2], self.num_threads)
        return output_x, dz_dsigma
