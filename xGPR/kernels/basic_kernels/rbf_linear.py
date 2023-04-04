"""Describes the RBF + Linear kernel, a sum of an RBF kernel
and a Linear kernel. This is highly effective in cases where
there is a linear trend but with local deviations."""
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

class RBFLinear(KernelBaseclass, ABC):
    """An implementation of a Linear kernel + an RBF kernel.
    Since it inherits from KernelBaseclass, it includes the
    attributes of that class. Only additional attributes unique
    to this class are described here.

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
        internal_rffs (int): The number of random features that will be generated.
            This is different than num_rffs (from the parent class), which is what
            the kernel will report to anyone asking how many features it generates.
            The reason for this difference is that the linear + rbf kernel
            concatenates the input (for the linear portion of the kernel) to
            the random features generated for the RBF portion.
        num_freqs (int): The number of frequencies to sample. Note that this is
            calculated based on internal_rffs not num_rffs so the calculation
            performed by the parent class is overriden.
        radem_diag: The diagonal matrices for the SORF transform. Type is int8.
        chi_arr: A diagonal array whose elements are drawn from the chi
            distribution. Ensures the marginals of the matrix resulting
            from S H D1 H D2 H D3 are correct.
        feature_gen: A reference to the wrapped C function that generates
            features.
        gradfun: A reference to the wrapped C function that calculates
            gradients.
    """

    def __init__(self, xdim, num_rffs, random_seed = 123, device = "cpu",
                num_threads = 2, double_precision = False, **kwargs):
        """Constructor. Calls the KernelBaseclass
        constructor first.

        Args:
            num_rffs (int): The user-requested number of random Fourier features.
                For sine-cosine kernels (RBF, Matern), this will be saved by the
                class as num_rffs. Note that the number of input features (from xdim)
                is subtracted from this to generate internal_rffs. If the result is
                less than zero, the kernel will generate an exception upon being
                created.
            xdim (tuple): The dimensions of the input. Either (N, D) or (N, M, D)
                where N is the number of datapoints, D is number of features
                and M is number of timepoints or sequence elements (convolution
                kernels only).
            num_threads (int): The number of threads to use for generating random
                features if running on CPU. If running on GPU, this is ignored.
            random_seed (int): The seed to the random number generator.
            double_precision (bool): If True, generate random features in double precision.
                Otherwise, generate as single precision.

        Raises:
            ValueError: If a non 2d input array dimensionality is supplied.
        """
        if len(xdim) != 2:
            raise ValueError("The dimensionality of the input is inappropriate for "
                        "the kernel you have selected.")

        #Although this IS a sine_cosine kernel, we don't want the parent class
        #to enforce that num_rffs be a multiple of two (only needs to be true
        #for internal rffs), so we set sine_cosine_kernel to False.
        super().__init__(num_rffs, xdim, num_threads = num_threads,
                sine_cosine_kernel = False,
                double_precision = double_precision)

        self.internal_rffs = num_rffs - xdim[1]
        if self.internal_rffs <= 1 or not (self.internal_rffs / 2).is_integer():
            raise ValueError("For the RBFLinear kernel, the number of 'random' "
                    "features requested includes the number of features in "
                    "the input. So, for example, if the input is a length 100 "
                    "vector and training_rffs is 1000, 900 random features will "
                    "be generated and the input features will be concatenated to "
                    "this to yield 1000 'random' features. The number of "
                    "training and fitting rffs requested should therefore be "
                    "at least num_input_features + 2, and after the input length "
                    "is subtracted, the remainder should be a power of two. The number of "
                    "variance_rffs requested is not affected.")
        self.hyperparams = np.ones((4))
        self.bounds = np.asarray([[1e-3,1e1], [0.1, 10], [1e-2, 15], [1e-6, 1e2]])


        self.num_freqs = int(self.internal_rffs / 2)
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
        self.device = device



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
        xtrans[:,:,:self.xdim[1]] = input_x[:,None,:] * self.hyperparams[3]

        output_x = self.empty((input_x.shape[0], self.num_rffs), self.out_type)
        random_features = self.empty((input_x.shape[0], self.internal_rffs),
                        self.out_type)
        self.feature_gen(xtrans, random_features, self.radem_diag, self.chi_arr,
                self.hyperparams[1], self.num_threads)

        output_x[:,:self.internal_rffs] = random_features
        output_x[:,self.internal_rffs:] = input_x * self.hyperparams[2] * self.hyperparams[1]
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
        random_features = self.empty((input_x.shape[0], self.internal_rffs),
                self.out_type)
        output_x = self.empty((input_x.shape[0], self.num_rffs), self.out_type)
        output_grad = self.zero_arr((input_x.shape[0], self.num_rffs, 2), self.out_type)

        output_grad[:,:self.internal_rffs,1:2] = self.gradfun(xtrans, random_features,
                        self.radem_diag, self.chi_arr, self.hyperparams[1],
                        self.hyperparams[3], self.num_threads)

        output_grad[:,self.internal_rffs:,0] = input_x * self.hyperparams[1]

        output_x[:,:self.internal_rffs] = random_features
        output_x[:,self.internal_rffs:] = input_x * self.hyperparams[2] * self.hyperparams[1]
        return output_x, output_grad
