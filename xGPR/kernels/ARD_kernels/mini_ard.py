"""A 'mini-ARD' kernel that assigns one lengthscale per
feature category for different feature types.
Accepts only fixed vectors as input."""
from math import ceil

import numpy as np
from scipy.stats import chi

from xGPR.xgpr_cpu_rfgen_cpp_ext import cpuFastHadamardTransform2D as dFHT2d
from xGPR.xgpr_cpu_rfgen_cpp_ext import cpuRBFFeatureGen, cpuMiniARDGrad
try:
    import cupy as cp
    from xGPR.xgpr_cuda_rfgen_cpp_ext import cudaRBFFeatureGen, cudaMiniARDGrad
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
        hyperparams (np.ndarray): A length one + number of feature
            classes array of hyperparameters: lambda_ (noise)
            and the remainder are inverse lengthscales.
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
                device = "cpu", double_precision = False,
                kernel_spec_parms = {}):
        """Constructor.

        Args:
            xdim (tuple): The dimensions of the input.
            num_rffs (int): The user-requested number of random Fourier features.
            random_seed (int): The seed to the random number generator.
            device (str): One of 'cpu', 'cuda'. Indicates the starting device.
            double_precision (bool): Whether to use double precision for the FHT
                when generating random features. It is generally best to set to
                False -- setting to True increases computational cost for negligible
                benefit.
            kernel_spec_parms (dict): A dictionary of kernel-specific parameters.
                In this case, must contain "split_points", which is a list
                of split points across input arrays.

        Raises:
            ValueError: A ValueError is raised if the needed kernel_spec_parms
                are not supplied, or if the split points supplied are invalid.
        """
        super().__init__(num_rffs, xdim, sine_cosine_kernel = True,
                double_precision = double_precision,
                kernel_spec_parms = kernel_spec_parms)
        if len(self._xdim) != 2:
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

        self.hyperparams = np.ones((self.split_pts.shape[0]))
        bounds = [[1e-3,1e2]] + [[1e-6, 1e2] for i in
                range(self.hyperparams.shape[0] - 1)]
        self.bounds = np.asarray(bounds)

        self.padded_dims = 2**ceil(np.log2(max(xdim[-1], 2)))

        radem_array = np.asarray([-1,1], dtype=np.int8)
        rng = np.random.default_rng(random_seed)
        if self.padded_dims < self.num_freqs:
            self.nblocks = ceil(self.num_freqs / self.padded_dims)
        else:
            self.nblocks = 1
        self.radem_diag = rng.choice(radem_array, size=(3, 1, self.nblocks * self.padded_dims),
                                replace=True)
        self.chi_arr = chi.rvs(df=self.padded_dims, size=self.num_freqs,
                            random_state = random_seed)


        #Converts the hyperparameters into a "full" array of the
        #same length as padded dims, just as if this were an ARD.
        #Also, store the positions that map to different hparams.
        self.full_ard_weights = np.zeros((xdim[-1]))
        self.ard_position_key = np.zeros((xdim[-1]), dtype=np.int32)

        if not self.double_precision:
            self.chi_arr = self.chi_arr.astype(np.float32)

        self.kernel_specific_set_hyperparams()

        self.precomputed_weights = None
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
        if n_splits < 1:
            raise ValueError("There must be at least one split point to use MiniARD.")
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
            if not isinstance(self.radem_diag, np.ndarray):
                if self.precomputed_weights is not None:
                    self.precomputed_weights = cp.asnumpy(self.precomputed_weights)
                self.radem_diag = cp.asnumpy(self.radem_diag)
                self.full_ard_weights = cp.asnumpy(self.full_ard_weights)
                self.chi_arr = cp.asnumpy(self.chi_arr)
                self.ard_position_key = cp.asnumpy(self.ard_position_key)
                self.chi_arr = cp.asnumpy(self.chi_arr)
        else:
            self.radem_diag = cp.asarray(self.radem_diag)
            self.chi_arr = cp.asarray(self.chi_arr)
            if self.precomputed_weights is not None:
                self.precomputed_weights = cp.asarray(self.precomputed_weights)
            self.full_ard_weights = cp.asarray(self.full_ard_weights)
            self.ard_position_key = cp.asarray(self.ard_position_key)


    def kernel_specific_set_hyperparams(self):
        """Once hyperparameters have been reset, this kernel needs
        to repopulate an array it will use when generating random
        features."""
        self.full_ard_weights[:] = 0
        self.ard_position_key[:] = 0
        for i in range(1, self.split_pts.shape[0]):
            self.full_ard_weights[self.split_pts[i-1]:self.split_pts[i]] = \
                    self.hyperparams[i]
            self.ard_position_key[self.split_pts[i-1]:self.split_pts[i]] = i - 1


    def kernel_specific_transform(self, input_x, sequence_length = None):
        """Generates random features for an input array.

        Args:
            x: Either a cupy or numpy array containing the input.
            sequence_length: Accepted for consistency with baseclass
                but not used by this kernel and thus ignored.

        Returns:
            xtrans: A cupy or numpy array containing the generated features.
        """
        xtrans = input_x * self.full_ard_weights[None,:]

        if self.device == "cpu":
            output_x = np.zeros((input_x.shape[0], self.num_rffs), np.float64)
            if not self.double_precision:
                xtrans = xtrans.astype(np.float32)
            cpuRBFFeatureGen(xtrans, output_x, self.radem_diag, self.chi_arr,
                    self.fit_intercept)
        else:
            output_x = cp.zeros((input_x.shape[0], self.num_rffs), cp.float64)
            if not self.double_precision:
                xtrans = xtrans.astype(cp.float32)
            cudaRBFFeatureGen(xtrans, output_x, self.radem_diag, self.chi_arr,
                    self.fit_intercept)
        return output_x


    def precompute_weights(self):
        """The kernel does not automatically generate precomputed weights,
        because for generating features during fitting or prediction,
        these are not necessary and would increase model size unnecessarily.
        Only during gradient calculations are they required. The first time
        the kernel is asked to generate a gradient, then, it will precompute
        and store weights.

        Note that precomputing weights is only helpful for ARD kernels
        because of the greater complexity of calculating the gradient
        using FHT only."""
        precomp_weights = []
        #Currently the FHT-2d only operation is only implemented for
        #CPU. Since the precompute weights operation is only performed
        #once, it has not been a high priority for optimization.
        #TODO: Add this for GPU, and transfer the operations under
        #this function to C / Cuda code.
        if self.device == "cuda":
            self.radem_diag = cp.asnumpy(self.radem_diag)
            self.chi_arr = cp.asnumpy(self.chi_arr)

        norm_constant = np.log2(self.padded_dims) / 2.0
        norm_constant = 1.0 / (2.0**norm_constant)

        padded_chi_arr = np.zeros((self.nblocks * self.padded_dims))
        padded_chi_arr[:self.chi_arr.shape[0]] = self.chi_arr

        for i in range(self.nblocks):
            ident_mat = np.eye(self.padded_dims)
            init_pt, cut_pt = i * self.padded_dims, (i+1) * self.padded_dims
            ident_mat *= self.radem_diag[0:1,0,init_pt:cut_pt] * norm_constant
            dFHT2d(ident_mat)
            ident_mat *= self.radem_diag[1:2,0,init_pt:cut_pt] * norm_constant
            dFHT2d(ident_mat)
            ident_mat *= self.radem_diag[2:3,0,init_pt:cut_pt] * norm_constant
            dFHT2d(ident_mat)


            ident_mat *= padded_chi_arr[init_pt:cut_pt]
            precomp_weights.append(ident_mat.T[:,:self._xdim[-1]])


        self.precomputed_weights = np.vstack(precomp_weights)[:self.num_freqs,:]
        if not self.double_precision:
            self.precomputed_weights = self.precomputed_weights.astype(np.float32)

        self.precomputed_weights = np.ascontiguousarray(self.precomputed_weights)
        if self.device == "cuda":
            self.radem_diag = cp.asarray(self.radem_diag)
            self.precomputed_weights = cp.asarray(self.precomputed_weights)
            self.chi_arr = cp.asarray(self.chi_arr)


    def kernel_specific_gradient(self, input_x, sequence_length = None):
        """The gradient for kernel-specific hyperparameters is calculated
        using an array (dz_dsigma) specific to each kernel.

        Args:
            input_x: A cupy or numpy array containing the raw input data.
            sequence_length: Accepted for consistency with baseclass
                but not used by this kernel and thus ignored.

        Returns:
            output_x: A cupy or numpy array containing the random feature
                representation of the input.
            dz_dsigma: A cupy or numpy array containing the derivative of
                output_x with respect to the kernel-specific hyperparameters.
        """
        if self.precomputed_weights is None:
            self.precompute_weights()
        max_map_position = int(self.ard_position_key.max())

        if self.device == "cpu":
            xtrans = np.zeros((input_x.shape[0], self.num_rffs), np.float64)
            dz_dsigma = np.zeros((input_x.shape[0], self.num_rffs,
                max_map_position + 1), np.float64)
            cpuMiniARDGrad(input_x, xtrans, self.precomputed_weights,
                self.ard_position_key, self.full_ard_weights,
                dz_dsigma, self.fit_intercept)
        else:
            xtrans = cp.zeros((input_x.shape[0], self.num_rffs), cp.float64)
            dz_dsigma = cp.zeros((input_x.shape[0], self.num_rffs,
                max_map_position + 1), cp.float64)
            cudaMiniARDGrad(input_x, xtrans, self.precomputed_weights,
                self.ard_position_key, self.full_ard_weights,
                dz_dsigma, self.fit_intercept)

        return xtrans, dz_dsigma
