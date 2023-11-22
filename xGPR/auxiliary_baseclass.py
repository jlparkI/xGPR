"""Describes the AuxiliaryBaseclass from which other model classes inherit.

The AuxiliaryBaseclass describes class attributes and methods shared by
auxiliary tools like kernel_xPCA and kernel_FGen.
"""
import sys

try:
    import cupy as cp
except:
    pass

from .kernels import KERNEL_NAME_TO_CLASS
from .constants import constants



class AuxiliaryBaseclass():
    """The base class for auxiliary toolkits in xGPR, primarily
    for performing kernel PCA and kernel KMeans.

    Attributes:
        kernel: The kernel object for the posterior predictive mean. The class of
            this object will depend on the kernel specified by the user.
        device (str): One of "gpu", "cpu". The user can update this as desired.
            All predict / tune / fit operations are carried out using the
            current device.
        verbose (bool): If True, regular updates are printed during
            hyperparameter tuning and fitting.
    """

    def __init__(self, num_rffs:int, hyperparams, dataset,
                    kernel_choice:str = "RBF", device:str = "cpu",
                    kernel_specific_params:dict = constants.DEFAULT_KERNEL_SPEC_PARMS,
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
            dataset: A valid dataset object.
            kernel_choice (str): The kernel that the model will use.
                Defaults to 'RBF'. Must be in kernels.kernel_list.
                KERNEL_NAME_TO_CLASS.
            device (str): Determines whether calculations are performed on
                'cpu' or 'gpu'. The initial entry can be changed later
                (i.e. model can be transferred to a different device).
                Defaults to 'cpu'.
            kernel_specific_params (dict): Contains kernel-specific parameters --
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
        kernel_specific_params["intercept"] = False

        self.verbose = verbose

        if kernel_choice not in KERNEL_NAME_TO_CLASS:
            raise ValueError("An unrecognized kernel choice was supplied.")
        self.kernel = KERNEL_NAME_TO_CLASS[kernel_choice](dataset.get_xdim(),
                            num_rffs, random_seed, device,
                            num_threads, double_precision_fht,
                            kernel_spec_parms = kernel_specific_params)
        self.device = device
        full_hparams = self.kernel.get_hyperparams()
        if full_hparams.shape[0] > 1:
            full_hparams[1:] = hyperparams
        self.kernel.set_hyperparams(full_hparams)


    def pre_prediction_checks(self, input_x):
        """Checks input data to ensure validity.

        Args:
            input_x (np.ndarray): A numpy array containing the input data.
            get_var (bool): Whether a variance calculation is desired.

        Returns:
            x_array: A cupy array (if self.device is gpu) or a reference
                to the unmodified input array otherwise.

        Raises:
            ValueError: If invalid inputs are supplied,
                a detailed ValueError is raised to explain.
        """
        x_array = input_x
        if not self.kernel.validate_new_datapoints(input_x):
            raise ValueError("The input has incorrect dimensionality.")
        if self.device == "gpu":
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
            x_array = cp.asarray(input_x)

        return x_array


    def transform_data(self, input_x, chunk_size:int = 2000):
        """Generate the random features for each chunk
        of an input array. This function is a generator
        so it will yield the random features as blocks
        of shape (chunk_size, fitting_rffs).

        Args:
            input_x (np.ndarray): The input data. Should be a 2d numpy
                array (if non-convolution kernel) or 3d (if convolution
                kernel).
            chunk_size (int): The number of datapoints to process at
                a time. Lower values limit memory consumption. Defaults
                to 2000.

        Yields:
            x_trans (array): An array containing the random features
                generated for a chunk of the input. Shape is
                (chunk_size, fitting_rffs).

        Raises:
            ValueError: If the dimensionality or type of the input does
                not match what is expected, or if the model has
                not yet been fitted, a ValueError is raised.
        """
        xdata = self.pre_prediction_checks(input_x)
        for i in range(0, xdata.shape[0], chunk_size):
            cutoff = min(i + chunk_size, xdata.shape[0])
            yield self.kernel.transform_x(xdata[i:cutoff, :])


    ####The remaining functions are all getters / setters.


    @property
    def device(self):
        """Property definition for the device attribute."""
        return self._device

    @device.setter
    def device(self, value):
        """Setter for the device attribute."""
        if value not in ["cpu", "gpu"]:
            raise ValueError("Device must be in ['cpu', 'gpu'].")

        if "cupy" not in sys.modules and value == "gpu":
            raise ValueError("You have specified the gpu fit mode but CuPy is "
                "not installed. Currently CPU only fitting is available.")

        if "cuda_rf_gen_module" not in sys.modules and value == "gpu":
            raise ValueError("You have specified the gpu fit mode but the "
                "cudaHadamardTransform module is not installed / "
                "does not appear to have installed correctly. "
                "Currently CPU only fitting is available.")

        self.kernel.device = value
        if value == "gpu":
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
        self._device = value
