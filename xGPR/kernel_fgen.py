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
                'cpu' or 'cuda'. The initial entry can be changed later
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


    def predict(self, input_x, sequence_lengths = None, chunk_size:int = 2000):
        """Generates random features for the input.

        Args:
            input_x (np.ndarray): A numpy array of xvalues for which RFFs will
                be generated.
            sequence_lengths: None if you are using a fixed-vector kernel (e.g.
                RBF) and a 1d array of the number of elements in each sequence /
                nodes in each graph if you are using a graph or Conv1d kernel.
            chunk_size (int): How many datapoints to process at a time. The
                results for all datapoints are returned simultaneously regardless --
                this parameter just affects memory consumption (smaller chunk_size
                = less memory) and speed (larger chunk_size = slightly faster).
        Returns:
            preds (np.ndarray): A 2d array of RFFs where the second dimension is
                the number of rffs.

        Raises:
            ValueError: A ValueError is raised if inappropriate inputs are
                supplied.
        """
        self.pre_prediction_checks(input_x, sequence_lengths)
        preds = []
        for i in range(0, input_x.shape[0], chunk_size):
            cutoff = min(i + chunk_size, input_x.shape[0])
            if sequence_lengths is not None:
                preds.append(self.kernel.transform_x(input_x[i:cutoff, ...],
                    sequence_lengths[i:cutoff]))
            else:
                preds.append(self.kernel.transform_x(input_x[i:cutoff, ...]))

        if self.device == "cuda":
            preds = cp.asnumpy(cp.vstack(preds))
        else:
            preds = np.vstack(preds)
        return preds
