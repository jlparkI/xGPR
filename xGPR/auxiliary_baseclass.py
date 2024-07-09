"""Describes the AuxiliaryBaseclass from which other model classes inherit.

The AuxiliaryBaseclass describes class attributes and methods shared by
auxiliary tools like kernel_xPCA and kernel_FGen.
"""
import sys
import numpy as np

try:
    import cupy as cp
except:
    pass

from .kernels import KERNEL_NAME_TO_CLASS, ARR_3D_KERNELS
from .constants import constants



class AuxiliaryBaseclass():
    """The base class for auxiliary toolkits in xGPR, primarily
    for performing kernel PCA and kernel KMeans.

    Attributes:
        kernel: The kernel object for the posterior predictive mean. The class of
            this object will depend on the kernel specified by the user.
        device (str): One of "cuda", "cpu". The user can update this as desired.
            All predict / tune / fit operations are carried out using the
            current device.
        double_precision_fht (bool): Indicates whether we are using double precision.
        verbose (bool): If True, regular updates are printed during
            hyperparameter tuning and fitting.
    """

    def __init__(self, num_rffs:int, hyperparams, num_features:int,
                    kernel_choice:str = "RBF", device:str = "cpu",
                    kernel_settings:dict = constants.DEFAULT_KERNEL_SPEC_PARMS,
                    random_seed:int = 123, verbose:bool = True,
                    num_threads:int = 2,
                    double_precision_fht:bool = False):
        """Constructor.

        Args:
            num_rffs (int): The number of random Fourier features
                to use for the auxiliary device.
            hyperparams (np.ndarray): A numpy array containing the kernel-specific
                hyperparameter. If you have fitted an xGPR model, the first
                hyperparameter is in general not kernel specific, so
                my_model.get_hyperparams()[1:] will retrieve the hyperparameters you
                need. For most kernels there is only one kernel-specific hyperparameter.
                For kernels with no kernel-specific hyperparameter (e.g. arc-cosine
                and polynomial kernels), this argument is ignored.
            num_features (int): The number of features (i.e. the expected length
                of the last dimension) of typical input.
            dataset: A valid dataset object.
            kernel_choice (str): The kernel that the model will use.
                Defaults to 'RBF'. Must be in kernels.kernel_list.
                KERNEL_NAME_TO_CLASS.
            device (str): Determines whether calculations are performed on
                'cpu' or 'cuda'. The initial entry can be changed later
                (i.e. model can be transferred to a different device).
                Defaults to 'cpu'.
            kernel_settings (dict): Contains kernel-specific parameters --
                e.g. 'matern_nu' for the nu for the Matern kernel, or 'conv_width'
                for the conv1d kernel.
            random_seed (int): A seed for the random number generator.
            verbose (bool): If True, regular updates are printed
                during fitting and tuning. Defaults to True.
            num_threads (int): The number of threads to use for random feature generation
                if running on CPU. If running on GPU, this argument is ignored.
            double_precision_fht (bool): If True, use double precision during FHT for
                generating random features. For most problems, it is not beneficial
                to set this to True -- it merely increases computational expense
                with negligible benefit -- but this option is useful for testing.
                Defaults to False.
        """
        #We should never use a y-intercept for kPCA or clustering.
        kernel_settings["intercept"] = False

        self.verbose = verbose

        if kernel_choice not in KERNEL_NAME_TO_CLASS:
            raise ValueError("An unrecognized kernel choice was supplied.")

        if kernel_choice in ARR_3D_KERNELS:
            if "conv_width" in kernel_settings:
                xdim = (1, kernel_settings["conv_width"], num_features)
            else:
                xdim = (1, 10, num_features)
        else:
            xdim = (1, num_features)

        self.kernel = KERNEL_NAME_TO_CLASS[kernel_choice](xdim,
                            num_rffs, random_seed, device,
                            num_threads, double_precision_fht,
                            kernel_spec_parms = kernel_settings)

        self.double_precision_fht = double_precision_fht
        self.device = device
        full_hparams = self.kernel.get_hyperparams()
        if full_hparams.shape[0] > 1:
            full_hparams[1:] = hyperparams
        self.kernel.set_hyperparams(full_hparams)


    def pre_prediction_checks(self, input_x, sequence_lengths):
        """Checks input data to ensure validity.

        Args:
            input_x (np.ndarray): A numpy array containing the input data.
            sequence_lengths: None if you are using a fixed-vector kernel (e.g.
                RBF) and a 1d array of the number of elements in each sequence /
                nodes in each graph if you are using a graph or Conv1d kernel.

        Returns:
            x_array: A cupy array (if self.device is cuda) or a reference
                to the unmodified input array otherwise.

        Raises:
            ValueError: If invalid inputs are supplied,
                a detailed ValueError is raised to explain.
        """
        if not self.kernel.validate_new_datapoints(input_x):
            raise ValueError("The input has incorrect dimensionality.")
        if sequence_lengths is None:
            if len(input_x.shape) != 2:
                raise ValueError("sequence_lengths is required if using a "
                        "convolution kernel.")
        else:
            if len(input_x.shape) == 2:
                raise ValueError("sequence_lengths must be None if using a "
                    "fixed vector kernel.")

        if self.device == "cuda":
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()


    ####The remaining functions are all getters / setters.


    @property
    def device(self):
        """Property definition for the device attribute."""
        return self._device

    @device.setter
    def device(self, value):
        """Setter for the device attribute."""
        if value not in ["cpu", "cuda"]:
            raise ValueError("Device must be in ['cpu', 'cuda'].")

        if "cupy" not in sys.modules and value == "cuda":
            raise ValueError("You have specified the cuda fit mode but CuPy is "
                "not installed. Currently CPU only fitting is available.")

        if "xGPR.xgpr_cuda_rfgen_cpp_ext" not in sys.modules and value == "cuda":
            raise ValueError("You have specified the cuda fit mode but the "
                "cudaHadamardTransform module is not installed / "
                "does not appear to have installed correctly. "
                "Currently CPU only fitting is available.")

        self.kernel.device = value
        if value == "cuda":
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
        self._device = value
