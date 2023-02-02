"""Describes the SORFKernelBaseclass, which is used by the main
fixed-vector kernels -- RBF, Matern, Arccosine, ERF-NN.

All of these share some methods and attributes in common
which are stored here to avoid redundancy, mainly
pretransform_x and the initialization process."""
from abc import ABC
from math import ceil

import numpy as np
from scipy.stats import chi
from cpu_basic_hadamard_operations import doubleCpuSORFTransform as dSORF
from cpu_basic_hadamard_operations import floatCpuSORFTransform as fSORF

try:
    import cupy as cp
    from cuda_basic_hadamard_operations import doubleCudaPySORFTransform as dCudaSORF
    from cuda_basic_hadamard_operations import floatCudaPySORFTransform as fCudaSORF
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
        num_threads (int): The number of threads for CPU-based FHT.
        sorf_transform: A reference to the Cython function that wraps the
            C code which generates the random features.
        cosfunc: A convenience reference to either cp.cos or np.cos,
            as appropriate for current device.
        sinfunc: A convenience reference to either cp.sin or np.sin,
            as appropriate for current device.
    """

    def __init__(self, num_rffs, xdim, double_precision = True,
                            sine_cosine_kernel = False, random_seed = 123):
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
            random_seed (int): The seed to the random number generator.
            double_precision (bool): If True, generate random features in double precision.
                Otherwise, generate as single precision.
            sine_cosine_kernel (bool): If True, the kernel is a sine-cosine kernel,
                meaning it will sample self.num_freqs frequencies and use the sine
                and cosine of each to generate twice as many features
                (self.num_rffs). sine-cosine kernels only accept num_rffs
                that are even numbers.

        Raises:
            ValueError: If a non 2d input array dimensionality is supplied.
        """
        super().__init__(num_rffs, xdim, double_precision, sine_cosine_kernel)
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
        self.num_threads = 2

        self.sinfunc = None
        self.cosfunc = None
        self.sorf_transform = dSORF


    def pretransform_x(self, input_x):
        """Random feature generation for this class
        is divided into two steps. The first is
        the SORF transform, here called 'pretransform', which does not
        involve kernel hyperparameters and therefore is not kernel-specific.
        This first step is performed by this function.

        Args:
            input_x: A cupy or numpy array depending on self.device
                containing the input data.

        Returns:
            output_x: A cupy or numpy array depending on self.device
                containing the results of the SORF operation. Note
                that num_freqs rffs are generated, not num_rffs.
        """
        output_x = self.zero_arr((input_x.shape[0], self.nblocks, self.padded_dims),
                            dtype = self.dtype)
        output_x[:,:,:self.xdim[1]] = input_x[:,None,:]
        self.sorf_transform(output_x, self.radem_diag, self.num_threads)
        output_x = output_x.reshape((output_x.shape[0], output_x.shape[1] *
                        output_x.shape[2]))[:,:self.num_freqs]
        output_x *= self.chi_arr[None,:]
        return output_x



    def kernel_specific_set_device(self, new_device):
        """Called by parent class when device is changed. Moves
        some of the object parameters to the appropriate device
        and updates self.cosfunc, self.sinfunc, which are
        convenience references to np.cos / np.sin or cp.cos
        / cp.sin."""
        if new_device == "cpu":
            self.cosfunc = np.cos
            self.sinfunc = np.sin
            if self.double_precision:
                self.sorf_transform = dSORF
            else:
                self.sorf_transform = fSORF
            if not isinstance(self.radem_diag, np.ndarray):
                self.radem_diag = cp.asnumpy(self.radem_diag)
                self.chi_arr = cp.asnumpy(self.chi_arr).astype(self.dtype)
        else:
            if self.double_precision:
                self.sorf_transform = dCudaSORF
            else:
                self.sorf_transform = fCudaSORF
            self.cosfunc = cp.cos
            self.sinfunc = cp.sin
            self.radem_diag = cp.asarray(self.radem_diag)
            self.chi_arr = cp.asarray(self.chi_arr).astype(self.dtype)


    def finish_transform(self, input_x):
        """Random feature generation is divided into two steps. The
        first is shared between all SORF fixed-vector kernels and
        involves the structured orthogonal random features or SORF
        operation. The second is kernel-specific and involves the
        kernel's activation function. This second step is performed
        by finish_transform.

        Args:
            input_x: Either a cupy or numpy array containing the input.

        Returns:
            xtrans: A cupy or numpy array containing the generated features.
        """
        xtrans = self.empty((input_x.shape[0], self.num_rffs), self.out_type)
        xtrans[:,:input_x.shape[1]] = self.cosfunc(input_x * self.hyperparams[2])
        xtrans[:,input_x.shape[1]:] = self.sinfunc(input_x * self.hyperparams[2])
        xtrans *= (self.hyperparams[1] * np.sqrt(2 / self.num_rffs))
        return xtrans


    def transform_x(self, input_x):
        """Combines the two steps involved in random feature generation
        to generate random features.

        Args:
            input_x: Either a cupy or numpy array containing the input.

        Returns:
            xtrans: A cupy or numpy array containing the generated features.
        """
        xtrans = self.pretransform_x(input_x)
        return self.finish_transform(xtrans)



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
        input_x = self.pretransform_x(input_x)
        output_x = self.finish_transform(input_x)
        dz_dsigma = self.empty((input_x.shape[0], self.num_rffs, 1),
                                    dtype = self.out_type)
        dz_dsigma[:,:input_x.shape[1], 0] = -output_x[:,input_x.shape[1]:] * input_x
        dz_dsigma[:,input_x.shape[1]:, 0] = output_x[:,:input_x.shape[1]] * input_x
        return output_x, dz_dsigma
