"""This tool enables easy generation of random features for
approximate kernel k-means clustering or for other tasks."""
import numpy as np
try:
    import cupy as cp
except:
    print("CuPy not detected. xGPR will run in CPU-only mode.")

from .constants import constants
from .auxiliary_baseclass import AuxiliaryBaseclass



class KernelFGen(AuxiliaryBaseclass):
    """A tool for generating random features using
    a selected kernel, typically for use in e.g.
    kernel k-means clustering."""

    def __init__(self, num_rffs:int, hyperparams, num_features:int,
            kernel_choice:str = "RBF", device:str = "cpu",
            kernel_settings:dict = constants.DEFAULT_KERNEL_SPEC_PARMS,
            random_seed:int = 123, verbose:bool = True, num_threads:int = 2):
        """The constructor.

        Args:
            num_rffs (int): The number of random Fourier features
                to use for the auxiliary device.
            hyperparams (np.ndarray): A numpy array containing the kernel-specific
                hyperparameter. If you have fitted an xGPR model, the first
                hyperparameter is in general not kernel specific, so
                my_model.get_hyperparams()[1:] will retrieve the hyperparameters you
                need. For most kernels there is only one kernel-specific hyperparameter.
                For kernels with no kernel-specific hyperparameter (e.g. polynomial
                kernels), this argument is ignored.
            num_features (int): The number of features in your input data. This
                should be the last dimension of a typical input array.
            kernel_choice (str): The kernel that the model will use.
                Must be in kernels.kernel_list.KERNEL_NAME_TO_CLASS.
            device (str): Determines whether calculations are performed on
                'cpu' or 'gpu'. The initial entry can be changed later
                (i.e. model can be transferred to a different device).
                Defaults to 'cpu'.
            kernel_settings (dict): Contains kernel-specific parameters --
                e.g. 'matern_nu' for the nu for the Matern kernel, or 'conv_width'
                for sequence / timeseries kernels.
            random_seed (int): A seed for the random number generator.
            verbose (bool): If True, regular updates are printed
                during fitting and tuning. Defaults to True.
            num_threads (int): The number of threads to use for random feature generation
                if running on CPU. If running on GPU, this argument is ignored.
        """
        super().__init__(num_rffs, hyperparams, num_features,
                        kernel_choice, device, kernel_settings,
                        random_seed, verbose, num_threads)


    def predict(self, input_x, chunk_size:int = 2000):
        """Generates random features for the input."""
        xdata = self.pre_prediction_checks(input_x)
        preds = []
        for i in range(0, xdata.shape[0], chunk_size):
            cutoff = min(i + chunk_size, xdata.shape[0])
            preds.append(self.kernel.transform_x(xdata[i:cutoff, :]))

        if self.device == "gpu":
            preds = cp.asnumpy(cp.vstack(preds))
        else:
            preds = np.vstack(preds)
        return preds
