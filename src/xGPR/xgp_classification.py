"""Describes the xGPClassification class.

The xGPClassification class provides the tools needed to fit a classification
model and make predictions for new datapoints. It inherits from
ModelBaseclass.
"""
import numpy as np
try:
    import cupy as cp
except:
    print("CuPy not detected. xGPR will run in CPU-only mode.")

from .fitting_toolkit.nonlinear_cg_toolkit import nonlinear_CG_classification
from .constants import constants
from .model_baseclass import ModelBaseclass




class xGPClassification(ModelBaseclass):
    """Approximate kernelized classification."""

    def __init__(self, num_rffs:int = 256,
            kernel_choice:str = "RBF",
            device:str = "cpu",
            kernel_settings:dict = constants.DEFAULT_KERNEL_SPEC_PARMS,
            verbose:bool = True,
            random_seed:int = 123):
        """The class constructor. Passes arguments onto
        the parent class constructor.

        Args:
            num_rffs (int): The number of random Fourier features
                to use.
            kernel_choice (str): The kernel that the model will use.
            device (str): Determines whether calculations are performed on
                'cpu' or 'cuda'. The initial entry can be changed later
                (i.e. model can be transferred to a different device).
                Defaults to 'cpu'.
            kernel_settings (dict): Contains kernel-specific parameters --
                e.g. 'matern_nu' for the nu for the Matern kernel, or 'conv_width'
                for the conv1d kernel.
            verbose (bool): If True, regular updates are printed
                during fitting and tuning. Defaults to True.
            random_seed (int): The seed to the random number generator.
        """
        # Note that we pass 0 as the second argument for variance rffs
        # since the classifier does not currently compute variance.
        if not isinstance(kernel_settings, dict):
            raise RuntimeError("kernel_settings must be a dict.")
        super().__init__(num_rffs, 0,
                        kernel_choice, device = device,
                        kernel_settings = kernel_settings,
                        verbose = verbose, random_seed = random_seed)
        self.is_regression = False



    def predict(self, input_x, sequence_lengths = None, chunk_size:int = 2000):
        """Generate a predicted value for each
        input datapoint.

        Args:
            input_x (np.ndarray): The input data. Should be a 2d numpy
                array (if non-convolution kernel) or 3d (if convolution
                kernel).
            sequence_lengths: None if you are using a fixed-vector kernel (e.g.
                RBF) and a 1d array of the number of elements in each sequence /
                nodes in each graph if you are using a graph or Conv1d kernel.
            chunk_size (int): The number of datapoints to process at
                a time. Lower values limit memory consumption. Defaults
                to 2000.

        Returns:
            predictions (np.ndarray): Numpy array of shape (N x M) for
                N datapoints and M possible classes, with the probability
                of each class. If binary classification, M is treated as 1.

        Raises:
            RuntimeError: If the dimesionality or type of the input does
                not match what is expected, or if the model has
                not yet been fitted, a RuntimeError is raised.
        """
        # Note that get_var is always false here, so last arg is false.
        self.pre_prediction_checks(input_x, sequence_lengths, False)
        if self.gamma is None:
            raise RuntimeError("Model has not been fitted yet.")
        preds = []

        for i in range(0, input_x.shape[0], chunk_size):
            cutoff = min(i + chunk_size, input_x.shape[0])
            if sequence_lengths is not None:
                xfeatures = self.kernel.transform_x(input_x[i:cutoff,...],
                        sequence_lengths[i:cutoff])
            else:
                xfeatures = self.kernel.transform_x(input_x[i:cutoff,...])

            pred = xfeatures @ self.weights + self.gamma[None,:]

            # Numerically stable softmax. 2.71828 is e.
            pred -= pred.max(axis=1)[:,None]
            pred = 2.71828**pred
            pred /= pred.sum(axis=1)[:,None]

            preds.append(pred)

        if self.device == "cuda":
            return cp.asnumpy(cp.vstack(preds))
        return np.vstack(preds)



    def fit(self, dataset, preconditioner = None,
            tol:float = 1e-3,
            max_iter:int = 500,
            max_rank:int = 3000,
            min_rank:int = 512,
            autoselect_target_ratio:float = 30.,
            always_use_srht2:bool = False,
            run_diagnostics:bool = False):
        """Fits the model after checking that the input data
        is consistent with the kernel choice and other user selections.

        Args:
            dataset: Object of class OnlineDataset or OfflineDataset.
            preconditioner: Either None (default) or a valid Preconditioner 
                (generated by a call to build_preconditioner).
                If None and mode is 'cg', a preconditioner is
                automatically constructed at a max_rank chosen using several
                passes over the dataset. If mode is 'exact', this argument is
                ignored.
            tol (float): The threshold below which iterative strategies (CG)
                are deemed to have converged. Defaults to 1e-5. Note that how
                reaching the threshold is assessed may depend on the algorithm.
            max_iter (int): The maximum number of epochs for iterative strategies.
            max_rank (int): The largest size to which the preconditioner can be set.
                Ignored if not autoselecting a preconditioner (i.e. if
                mode is 'cg' and preconditioner = None). The default is
                significantly more than should be needed the vast majority of
                the time.
            min_rank (int): The smallest rank to which the preconditioner can be
                set. Ignored if not autoselecting a preconditioner (e.g. if mode is
                not 'cg' or preconditioner is not None). The default is usually fine.
                Consider setting to a smaller number if you always want to use the
                smallest preconditioner possible.
            autoselect_target_ratio (float): The target ratio if choosing the
                preconditioner size via autoselect. Lower values reduce the number
                of iterations needed to fit but increase preconditioner expense.
                Default is usually fine if 40 - 50 iterations are considered
                acceptable.
            always_use_srht2 (bool): If True, always use srht2 for preconditioner
                construction. This will reduce the number of iterations for
                fitting by 30% on average but will increase the time cost
                of preconditioner construction about 150%.
            run_diagnostics (bool): If True, some performance metrics are returned.

        Returns:
            Does not return anything unless run_diagnostics is True.
            n_iter (int): The number of iterations if applicable.
            losses (list): The loss on each iteration. Only for CG, otherwise,
                empty list.

        Raises:
            RuntimeError: The input dataset is checked for validity before tuning is
                initiated, an error is raised if problems are found."""
        self._run_pre_fitting_prep(dataset)
        self.weights = None
        self.n_classes = int(dataset.get_n_classes())
        losses = []

        if self.verbose:
            print("starting fitting")


        if preconditioner is None:
            preconditioner = self._autoselect_preconditioner(dataset,
                    min_rank = min_rank, max_rank = max_rank,
                    ratio_target = autoselect_target_ratio,
                    always_use_srht2 = always_use_srht2)
        cg_operator = nonlinear_CG_classification(dataset, self.kernel,
                self.device, self.verbose, preconditioner)

        self.weights, n_iter, losses = cg_operator.fit_model(max_iter, tol)
        if self.device == "cuda":
            self.gamma = cp.zeros((self.n_classes))
        else:
            self.gamma = np.zeros((self.n_classes))
        if self.verbose:
            print(f"CG iterations: {n_iter}")



        if self.verbose:
            print("Fitting complete.")
        if self.device == "cuda":
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()

        if run_diagnostics:
            return n_iter, losses
