"""Describes the xGPClassification class.

The xGPClassification class provides the tools needed to fit a regression
model and make predictions for new datapoints. It inherits from
ModelBaseclass.
"""
try:
    import cupy as cp
except:
    print("CuPy not detected. xGPR will run in CPU-only mode.")
import numpy as np

from .constants import constants
from .model_baseclass import ModelBaseclass

from .fitting_toolkit.lbfgs_fitting_toolkit import lBFGSModelFit




class xGPClassification(ModelBaseclass):
    """A subclass of GPModelBaseclass that houses methods
    unique to classification problems. Only attributes unique
    to this child are described here.

    Attributes:
        n_classes (int): The number of classes expected in the
            data. This is initialized when fit is called.
    """

    def __init__(self, num_rffs = 256,
                    kernel_choice="RBF",
                    device = "cpu",
                    kernel_specific_params = constants.DEFAULT_KERNEL_SPEC_PARMS,
                    verbose = True,
                    num_threads = 2,
                    random_seed = 123):
        """The class constructor. Passes arguments onto
        the parent class constructor.

        Args:
            num_rffs (int): The number of random Fourier features
                to use.
            kernel_choice (str): The kernel that the model will use.
            device (str): Determines whether calculations are performed on
                'cpu' or 'gpu'. The initial entry can be changed later
                (i.e. model can be transferred to a different device).
                Defaults to 'cpu'.
            kernel_specific_params (dict): Contains kernel-specific parameters --
                e.g. 'matern_nu' for the nu for the Matern kernel, or 'conv_width'
                for the conv1d kernel.
            verbose (bool): If True, regular updates are printed
                during fitting and tuning. Defaults to True.
            num_threads (int): The number of threads to use for random feature generation
                if running on CPU. If running on GPU, this argument is ignored.
            random_seed (int): The seed to the random number generator. Used
                throughout.
        """
        # Note that we pass 0 as the second argument for variance rffs
        # since the classifier does not currently compute variance.
        super().__init__(num_rffs, 0,
                        kernel_choice, device = device,
                        kernel_specific_params = kernel_specific_params,
                        verbose = verbose, num_threads = num_threads,
                        random_seed = random_seed)
        self.n_classes = 1



    def predict(self, input_x, chunk_size = 2000):
        """Generate a predicted value for each
        input datapoint.

        Args:
            input_x (np.ndarray): The input data. Should be a 2d numpy
                array (if non-convolution kernel) or 3d (if convolution
                kernel).
            chunk_size (int): The number of datapoints to process at
                a time. Lower values limit memory consumption. Defaults
                to 2000.

        Returns:
            predictions (np.ndarray): Numpy array of shape (N x M) for
                N datapoints and M possible classes, with the probability
                of each class. If binary classification, M is treated as 1.

        Raises:
            ValueError: If the dimesionality or type of the input does
                not match what is expected, or if the model has
                not yet been fitted, a ValueError is raised.
        """
        xdata = self.pre_prediction_checks(input_x, get_var = False)
        preds = []

        for i in range(0, xdata.shape[0], chunk_size):
            cutoff = min(i + chunk_size, xdata.shape[0])
            xfeatures = self.kernel.transform_x(xdata[i:cutoff, :])
            pred = xfeatures @ self.weights

            # Numerically stable softmax and sigmoid. 2.71828 is e.
            # The cutpoint of 20 on the sigmoid is arbitrary but
            # any large choice will give the same result.
            if self.n_classes == 2:
                pred = (1 / (1 + 2.71828**(pred.clip(max=30)))).flatten()
            else:
                pred -= pred.max(axis=1)[:,None]
                pred = 2.71828**pred
                pred /= pred.sum(axis=1)[:,None]

            preds.append(pred)

        if self.device == "gpu":
            return cp.asnumpy(cp.concatenate(preds))
        return preds



    def fit(self, dataset, tol = 1e-6,
                preset_hyperparams=None, max_iter = 500,
                run_diagnostics = False,
                mode = "lbfgs"):
        """Fits the model after checking that the input data
        is consistent with the kernel choice and other user selections.

        Args:
            dataset: Object of class OnlineDataset or OfflineDataset.
            tol (float): The threshold below which iterative strategies (L-BFGS, CG,
                SGD) are deemed to have converged. Defaults to 1e-5. Note that how
                reaching the threshold is assessed may depend on the algorithm.
            preset_hyperparams: Either None or a numpy array. If None,
                hyperparameters must already have been tuned using one
                of the tuning methods (e.g. tune_hyperparams_bayes_bfgs).
                If supplied, must be a numpy array of shape (N, 2) where
                N is the number of hyperparams for the kernel in question.
            max_iter (int): The maximum number of epochs for iterative strategies.
            random_seed (int): The random seed for the random number generator.
            run_diagnostics (bool): If True, the number of conjugate
                gradients and the preconditioner diagnostics ratio are returned.
            mode (str): Must be one of "cg", "lbfgs", "exact".
                Determines the approach used. If 'exact', self.kernel.get_num_rffs
                must be <= constants.constants.MAX_CLOSED_FORM_RFFS.

        Returns:
            Does not return anything unless run_diagnostics is True.
            n_iter (int): The number of iterations if applicable.
            losses (list): The loss on each iteration. Only for SGD and CG, otherwise,
                empty list.

        Raises:
            ValueError: The input dataset is checked for validity before tuning is
                initiated, an error is raised if problems are found."""
        self._run_pre_fitting_prep(dataset, preset_hyperparams)
        self.weights = None
        self.n_classes = dataset.max_class + 1

        if self.verbose:
            print("starting fitting")


        if mode == "lbfgs":
            model_fitter = lBFGSModelFit(dataset, self.kernel,
                    self.device, self.verbose, task_type = "classification",
                    n_classes = self.n_classes)
            self.weights, n_iter, losses = model_fitter.fit_model_lbfgs(max_iter, tol)


        else:
            raise ValueError("Unrecognized fitting mode supplied. Must provide one of "
                        "'lbfgs'.")


        if self.verbose:
            print("Fitting complete.")
        if self.device == "gpu":
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
        self._run_post_fitting_cleanup(dataset)

        if run_diagnostics:
            return n_iter, losses
