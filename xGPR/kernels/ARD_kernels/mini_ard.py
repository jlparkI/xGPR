"""A 'mini-ARD' kernel that assigns one lengthscale per
feature category for different feature types.
Accepts only fixed vectors as input."""
from math import ceil

import numpy as np
from scipy.stats import chi
from cpu_basic_hadamard_operations import doubleCpuSORFTransform as dSORF
from cpu_basic_hadamard_operations import floatCpuSORFTransform as fSORF
from cpu_basic_hadamard_operations import doubleCpuRBFFeatureGen as dRBF
from cpu_basic_hadamard_operations import floatCpuRBFFeatureGen as fRBF

try:
    import cupy as cp
    from cuda_basic_hadamard_operations import doubleCudaPySORFTransform as dCudaSORF
    from cuda_basic_hadamard_operations import floatCudaPySORFTransform as fCudaSORF
    from cuda_basic_hadamard_operations import doubleCudaRBFFeatureGen as dCudaRBF
    from cuda_basic_hadamard_operations import floatCudaRBFFeatureGen as fCudaRBF
except:
    pass
from ..kernel_baseclass import KernelBaseclass


class MiniARD(KernelBaseclass):
    """The MiniARD is a subset of automatic relevance determination
    (ARD), which assigns a lengthscale to every feature. MiniARD
    rather assigns a different lengthscale to each group of features
    (e.g. the first 20, the next 20 and so on), as specified by
    a user-provided list of "split points", points at which to split
    up the feature vector.

    This class inherits from KernelBaseclass. Only attributes unique
    to this child are described in this docstring.

    Attributes:
        hyperparams (np.ndarray): A length two + number of feature
            classes array of hyperparameters: lambda_ (noise),
            beta_ (amplitude) and the remainder are inverse lengthscales.
        cosfunc: A convenience reference to either cp.cos or np.cos,
            as appropriate for current device.
        sinfunc: A convenience reference to either cp.sin or np.sin,
            as appropriate for current device.
        split_pts (list): The points at which to split the
            input data; a list of up to length 3. I.e.,
            if x is the input data, lengthscale 1 applies
            to x[:,0:split_pts[0]], lengthscale 2 applies
            to x[:,split_pts[0]:split_pts[1]], etc.
        full_ard_weights (array): A cupy or numpy array (depending on device)
            containing the ARD weights for each input feature. This is
            repopulated whenever the hyperparameters are updated.
    """

    def __init__(self, xdim, num_rffs, random_seed = 123,
                device = "cpu", double_precision = False, kernel_spec_parms = {}):
        """Constructor for RBF.

        Args:
            xdim (tuple): The dimensions of the input.
            num_rffs (int): The user-requested number of random Fourier features.
            random_seed (int): The seed to the random number generator.
            device (str): One of 'cpu', 'gpu'. Indicates the starting device.
            kernel_spec_parms (dict): A dictionary of kernel-specific parameters.
                In this case, must contain "split_points", which is a list
                of split points across input arrays.

        Raises:
            ValueError: A ValueError is raised if the needed kernel_spec_parms
                are not supplied, or if the split points supplied are invalid.
        """
        super().__init__(num_rffs, xdim, random_seed, sine_cosine_kernel = True)
        if len(self.xdim) != 2:
            raise ValueError("The dimensionality of the input is inappropriate for "
                        "the kernel you have selected.")
        if "split_points" not in kernel_spec_parms:
            raise ValueError("For the MiniARD kernel, 'kernel_specific_params' "
                    "must contain a list called 'split_points'.")
        if not isinstance(kernel_spec_parms["split_points"], list):
            raise ValueError("For the MiniARD kernel, 'split_points' must "
                    "be a list.")
        self.split_pts = np.sort([0] + kernel_spec_parms["split_points"] + [xdim[1]])
        self.check_split_points(xdim)

        self.hyperparams = np.ones((1 + self.split_pts.shape[0]))
        bounds = [[1e-3,1e1], [0.1, 10]] + [[1e-6, 1e2] for i in
                range(self.hyperparams.shape[0] - 2)]
        self.bounds = np.asarray(bounds)

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

        #Converts the hyperparameters into a "full" array of the
        #same length as padded dims, just as if this were an ARD.
        self.full_ard_weights = np.zeros((xdim[-1]))
        self.kernel_specific_set_hyperparams()

        self.feature_gen = fRBF
        self.sinfunc = None
        self.cosfunc = None
        self.num_threads = 2
        self.sorf_transform = fSORF
        self.device = device


    def check_split_points(self, xdim):
        """Called during initialization to ensure the split points supplied
        by user are valid, or at any time split points are updated.

        Args:
            xdim (tuple): The shape of the anticipated input data.

        Raises:
            ValueError: A ValueError is raised if the splits supplied
                by user are not valid.
        """
        n_splits = self.split_pts.shape[0] - 2
        if n_splits < 1 or n_splits > 3:
            raise ValueError("The MiniARD kernel currently accepts from 1 "
                    "to 3 split points; you have supplied either more or less.")
        if self.split_pts[0] < 0:
            raise ValueError("The first split point must be > 0.")
        if self.split_pts[-1] > xdim[1]:
            raise ValueError("The last split point must be < shape[1] of the "
                    "input data.")
        if np.diff(self.split_pts).min() == 0:
            raise ValueError("At least two of the split points supplied are "
                        "identical.")


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
                self.feature_gen = dRBF
            else:
                self.sorf_transform = fSORF
                self.feature_gen = fRBF
            if not isinstance(self.radem_diag, np.ndarray):
                self.radem_diag = cp.asnumpy(self.radem_diag)
                self.full_ard_weights = cp.asnumpy(self.full_ard_weights).astype(self.dtype)
                self.chi_arr = cp.asnumpy(self.chi_arr)
            self.chi_arr = self.chi_arr.astype(self.dtype)
        else:
            if self.double_precision:
                self.sorf_transform = dCudaSORF
                self.feature_gen = dCudaRBF
            else:
                self.sorf_transform = fCudaSORF
                self.feature_gen = fCudaRBF
            self.cosfunc = cp.cos
            self.sinfunc = cp.sin
            self.radem_diag = cp.asarray(self.radem_diag)
            self.chi_arr = cp.asarray(self.chi_arr).astype(self.dtype)
            self.full_ard_weights = cp.asarray(self.full_ard_weights).astype(self.dtype)



    def kernel_specific_set_hyperparams(self):
        """Once hyperparameters have been reset, this kernel needs
        to repopulate an array it will use when generating random
        features."""
        self.full_ard_weights[:] = 0
        for i in range(1, self.split_pts.shape[0]):
            self.full_ard_weights[self.split_pts[i-1]:self.split_pts[i]] = \
                    self.hyperparams[i + 1]

    def transform_x(self, input_x):
        """Generates random features for an input array.

        Args:
            x: Either a cupy or numpy array containing the input.

        Returns:
            xtrans: A cupy or numpy array containing the generated features.
        """
        xtrans = self.zero_arr((input_x.shape[0], self.nblocks, self.padded_dims),
                            dtype = self.dtype)
        xtrans[:,:,:self.xdim[1]] = (input_x * self.full_ard_weights[None,:])[:,None,:]

        output_x = self.empty((input_x.shape[0], self.num_rffs), self.out_type)
        self.feature_gen(xtrans, output_x, self.radem_diag, self.chi_arr,
                self.hyperparams[1], self.num_freqs, self.num_threads)
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
        #TODO: The gradient calculation as implemented here is inefficient.
        #We can make it much more efficient, especially for cases where
        #we just want the derivative @ a vector. Rewrite this.
        dz_dsigma = self.empty((input_x.shape[0], self.num_rffs,
                    self.split_pts.shape[0] - 1), dtype = self.dtype)
        x_sorf = self.zero_arr((input_x.shape[0], self.nblocks, self.padded_dims),
                            dtype = self.dtype)
        x_sorf[:,:,:self.xdim[1]] = (input_x * self.full_ard_weights[None,:])[:,None,:]

        self.sorf_transform(x_sorf, self.radem_diag, self.num_threads)
        x_feat = x_sorf.reshape((x_sorf.shape[0], x_sorf.shape[1] *
                        x_sorf.shape[2]))[:,:self.num_freqs]
        x_feat *= self.chi_arr[None,:]

        xtrans = self.empty((x_sorf.shape[0], self.num_rffs), self.dtype)
        xtrans[:,:self.num_freqs] = self.cosfunc(x_feat)
        xtrans[:,self.num_freqs:] = self.sinfunc(x_feat)

        for i in range(1, self.split_pts.shape[0]):
            x_sorf[:] = 0
            x_sorf[:, :, self.split_pts[i-1]:self.split_pts[i]] = \
                    input_x[:,None,self.split_pts[i-1]:self.split_pts[i]]
            self.sorf_transform(x_sorf, self.radem_diag, self.num_threads)
            x_feat[:] = x_sorf.reshape((x_sorf.shape[0], x_sorf.shape[1] *
                        x_sorf.shape[2]))[:,:self.num_freqs]
            x_feat *= self.chi_arr[None,:]
            dz_dsigma[:,:self.num_freqs, i - 1] = \
                    -xtrans[:,self.num_freqs:] * x_feat
            dz_dsigma[:,self.num_freqs:, i - 1] = \
                    xtrans[:,:self.num_freqs] * x_feat

        xtrans *= (self.hyperparams[1] * np.sqrt(2 / self.num_rffs))
        dz_dsigma *= (self.hyperparams[1] * np.sqrt(2 / self.num_rffs))
        return xtrans, dz_dsigma
