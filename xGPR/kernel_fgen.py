"""This tool enables easy generation of random features for
approximate kernel k-means clustering. For now, it's essentially
a light wrapper on the kernel classes. In future, we may add
a k-means implementation of our own within this class, so the
end user does not need to use say scikit-learn's implementation.
This is important since scikit-learn's implementation requires
the whole dataset to 'live' in memory, which is obviously
not convenient if a large number of random features is used.
For now, then, this is probably best used in conjunction with
a small number of random features."""
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

    def __init__(self, num_rffs, hyperparams, dataset,
            kernel_choice = "RBF", device = "cpu",
            kernel_specific_params = constants.DEFAULT_KERNEL_SPEC_PARMS,
            random_seed = 123, verbose = True, num_threads = 2,
            double_precision_fht = False):
        """The constructor.

        Args:
            num_rffs (int): The number of random Fourier features
                to use for the auxiliary device.
            dataset: A valid dataset object.
            hyperparams (np.ndarray): A numpy array containing the kernel-specific
                hyperparameter. If you have fitted an xGPR model, the first two
                hyperparameters are general not kernel specific, so
                my_model.get_hyperparams()[2:] will retrieve the hyperparameters you
                need. For most kernels there is only one kernel-specific hyperparameter.
                For kernels with no kernel-specific hyperparameter (e.g. arc-cosine
                and polynomial kernels), this argument is ignored.
            kernel_choice (str): The kernel that the model will use.
                Must be in kernels.kernel_list.KERNEL_NAME_TO_CLASS.
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
        super().__init__(num_rffs, hyperparams, dataset,
                        kernel_choice, device, kernel_specific_params,
                        random_seed, verbose, num_threads,
                        double_precision_fht)


    def predict(self, input_x, chunk_size = 2000):
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
