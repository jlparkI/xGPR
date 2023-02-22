"""A 'mini-ARD' kernel that assigns one lengthscale per
feature category for different feature types.
Accepts only fixed vectors as input."""
import numpy as np
try:
    import cupy as cp
except:
    pass
from ..sorf_kernel_baseclass import SORFKernelBaseclass


class MiniARD(SORFKernelBaseclass):
    """The MiniARD is a subset of automatic relevance determination
    (ARD), which assigns a lengthscale to every feature. MiniARD
    rather assigns a different lengthscale to each group of features
    (e.g. the first 20, the next 20 and so on), as specified by
    a user-provided list of "split points", points at which to split
    up the feature vector.

    This class inherits from SORFKernelBaseclass which in turn inherits
    from KernelBaseclass. Only attributes unique to this child are
    described in this docstring.

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
    """

    def __init__(self, xdim, num_rffs, random_seed = 123,
                device = "cpu", kernel_spec_parms = {}):
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
        bounds = np.asarray(bounds)
        self.set_bounds(bounds, logspace=False)

        self.sinfunc = None
        self.cosfunc = None
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
            if not isinstance(self.radem_diag, np.ndarray):
                self.radem_diag = cp.asnumpy(self.radem_diag)
                self.chi_arr = cp.asnumpy(self.chi_arr).astype(self.dtype)
        else:
            self.cosfunc = cp.cos
            self.sinfunc = cp.sin
            self.radem_diag = cp.asarray(self.radem_diag)
            self.chi_arr = cp.asarray(self.chi_arr).astype(self.dtype)



    def pretransform_x(self, input_x):
        """Random feature generation is divided into two steps. The first is
        the SORF transform, here called 'pretransform'. For MOST kernels,
        this step does not involve hyperparameters and can therefore
        be shared. For this kernel, however, it DOES involve hyperparameters.
        Therefore we override the parent class pretransform_x here and
        merely return a reference to the input array (so we keep a
        consistent interface).

        Args:
            input_x: A cupy or numpy array depending on self.device
                containing the input data.

        Returns:
            output_x: A reference to the input array.
        """
        return input_x


    def finish_transform(self, input_x, multiply_by_beta = True):
        """Random feature generation is divided into two steps. The first is
        the SORF transform, here called 'pretransform'. For MOST kernels,
        this step does not involve hyperparameters and can therefore
        be shared. For this kernel, however, it DOES involve hyperparameters.
        For this kernel, therefore, the entirey of the random feature
        generation step is performed by finish_transform.

        Args:
            x: Either a cupy or numpy array containing the input.
            multiply_by_beta (bool): If False, do not multiply the
                result by beta (the amplitude). Used by gridsearch
                routines during optimization. Defaults to True.

        Returns:
            xtrans: A cupy or numpy array containing the generated features.
        """
        x_sorf = self.zero_arr((input_x.shape[0], self.nblocks, self.padded_dims),
                            dtype = self.dtype)
        x_sorf[:,:,:self.xdim[1]] = input_x[:,None,:]

        for i in range(1, self.split_pts.shape[0]):
            x_sorf[:, :, self.split_pts[i-1]:self.split_pts[i]] *= \
                    self.hyperparams[i + 1]
        self.sorf_transform(x_sorf, self.radem_diag, self.num_threads)
        x_sorf = x_sorf.reshape((x_sorf.shape[0], x_sorf.shape[1] *
                        x_sorf.shape[2]))[:,:self.num_freqs]
        x_sorf *= self.chi_arr[None,:]

        xtrans = self.empty((x_sorf.shape[0], self.num_rffs), self.dtype)
        xtrans[:,:self.num_freqs] = self.cosfunc(x_sorf)
        xtrans[:,self.num_freqs:] = self.sinfunc(x_sorf)
        if multiply_by_beta:
            xtrans *= (self.hyperparams[1] * np.sqrt(2 / self.num_rffs))
        else:
            xtrans *= np.sqrt(2 / self.num_rffs)
        return xtrans


    def kernel_specific_gradient(self, input_x, pre_sorf = False):
        """Since all kernels share the beta and lambda hyperparameters,
        the gradient for these can be calculated by the parent class.
        The gradient kernel-specific hyperparameters however is calculated
        using an array (dz_dsigma) specific to each
        kernel. The kernel-specific arrays are calculated here.

        Args:
            input_x: A cupy or numpy array containing the raw input data.
            pre_sorf (bool): If True, the input_x has already undergone
                the first step in random feature generation. This is only
                really applicable for the RBF and Matern kernels but
                this input is accepted here to preserve a common interface.

        Returns:
            output_x: A cupy or numpy array containing the random feature
                representation of the input.
            dz_dsigma: A cupy or numpy array containing the derivative of
                output_x with respect to the kernel-specific hyperparameters.
        """
        dz_dsigma = self.empty((input_x.shape[0], self.num_rffs,
                    self.split_pts.shape[0] - 1), dtype = self.dtype)
        x_sorf = self.zero_arr((input_x.shape[0], self.nblocks, self.padded_dims),
                            dtype = self.dtype)
        x_sorf[:,:,:self.xdim[1]] = input_x[:,None,:]

        for i in range(1, self.split_pts.shape[0]):
            x_sorf[:, :, self.split_pts[i-1]:self.split_pts[i]] *= \
                    self.hyperparams[i + 1]

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
