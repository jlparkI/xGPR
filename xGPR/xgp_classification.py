"""Describes the xGPDiscriminant class.

The xGPDiscriminant class provides the tools needed to fit a classification
model and make predictions for new datapoints. It inherits from
ModelBaseclass.
"""
try:
    import cupy as cp
    from .preconditioners.cuda_rand_nys_preconditioners import Cuda_RandNysPreconditioner
except:
    print("CuPy not detected. xGPR will run in CPU-only mode.")
import numpy as np
from .constants import constants
from .model_baseclass import ModelBaseclass

from .fitting_toolkit.cg_fitting_toolkit import cg_fit_lib_discriminant
from .fitting_toolkit.lbfgs_fitting_toolkit import lBFGSModelFit
from .fitting_toolkit.exact_fitting_toolkit import calc_discriminant_weights_exact
from .preconditioners.rand_nys_preconditioners import CPU_RandNysPreconditioner




class xGPDiscriminant(ModelBaseclass):
    """An approximate kernelized discriminant for classification.
    """

    def __init__(self, num_rffs:int = 256,
            kernel_choice:str = "RBF",
            device:str = "cpu",
            kernel_settings:dict = constants.DEFAULT_KERNEL_SPEC_PARMS,
            verbose:bool = True,
            num_threads:int = 2,
            random_seed:int = 123):
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
            kernel_settings (dict): Contains kernel-specific parameters --
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
                        kernel_settings = kernel_settings,
                        verbose = verbose, num_threads = num_threads,
                        random_seed = random_seed)



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
            ValueError: If the dimesionality or type of the input does
                not match what is expected, or if the model has
                not yet been fitted, a ValueError is raised.
        """
        xdata, sequence_lengths = self.pre_prediction_checks(input_x, sequence_lengths, False)
        if self._gamma is None:
            raise ValueError("Model has not been fitted yet.")
        preds = []

        for i in range(0, xdata.shape[0], chunk_size):
            cutoff = min(i + chunk_size, xdata.shape[0])
            if sequence_lengths is not None:
                xfeatures = self.kernel.transform_x(xdata[i:cutoff,...], sequence_lengths[i:cutoff])
            else:
                xfeatures = self.kernel.transform_x(xdata[i:cutoff,...])

            pred = xfeatures @ self.weights + self._gamma[None,:]

            # Numerically stable softmax. 2.71828 is e.
            pred -= pred.max(axis=1)[:,None]
            pred = 2.71828**pred
            pred /= pred.sum(axis=1)[:,None]

            preds.append(pred)

        if self.device == "gpu":
            return cp.asnumpy(cp.vstack(preds))
        return np.vstack(preds)


    def _get_x_mean(self, dataset):
        """Compiles the overall data mean. This is a faster operation
        than obtaining all of the targets (see get_targets below.)

        Args:
            dataset: A valid Dataset object.

        Returns:
            x_mean (ndarray): The mean of the random features across
                the whole dataset. Shape (num_rffs).
        """
        if self.device == "gpu":
            x_mean = cp.zeros((self.kernel.get_num_rffs()))
        else:
            x_mean = np.zeros((self.kernel.get_num_rffs()))

        for xdata, ldata in dataset.get_chunked_x_data():
            xfeatures = self.kernel.transform_x(xdata, ldata)
            x_mean += xfeatures.sum(axis=0)

        x_mean /= dataset.get_ndatapoints()
        return x_mean


    def _get_targets(self, dataset):
        """Compiles the class-specific means and the overall data
        mean for fitting.

        Args:
            dataset: A valid Dataset object.

        Returns:
            x_mean (ndarray): The mean of the random features across
                the whole dataset. Shape (num_rffs).
            targets (ndarray): The mean of the random features for
                datapoints in each class. Shape (num_rffs, n_classes).
        """
        self.n_classes = dataset.get_n_classes()

        #This procedure is a little clunky. TODO: Replace this with a
        #wrapped CUDA & CPU function to speed up this calculation.
        if self.device == "gpu":
            x_mean = cp.zeros((self.kernel.get_num_rffs()))
            targets = cp.zeros((self.kernel.get_num_rffs(), self.n_classes))
            n_pts_per_class = cp.zeros((self.n_classes))
            unique_elements = cp.unique
            ytype, locator = cp.int32, cp.where
        else:
            x_mean = np.zeros((self.kernel.get_num_rffs()))
            targets = np.zeros((self.kernel.get_num_rffs(), self.n_classes))
            n_pts_per_class = np.zeros((self.n_classes))
            unique_elements = np.unique
            ytype, locator = np.int32, np.where

        for (xdata, ydata, ldata) in dataset.get_chunked_data():
            ydata = ydata.astype(ytype)
            xfeatures = self.kernel.transform_x(xdata, ldata)
            if ydata.max() > self.n_classes or ydata.min() < 0:
                raise ValueError("Unexpected y-values encountered.")

            cts = unique_elements(ydata, return_counts=True)

            for i in cts[0].tolist():
                idx = locator(ydata==i)[0]
                targets[:,i] += xfeatures[idx,:].sum(axis=0)

            x_mean += xfeatures.sum(axis=0)
            n_pts_per_class[cts[0]] += cts[1]

        x_mean /= float(dataset.get_ndatapoints())
        targets /= n_pts_per_class[None,:]
        return x_mean, targets


    def build_preconditioner(self, dataset, max_rank:int = 512, method:str = "srht"):
        """Builds a preconditioner. The resulting preconditioner object
        can be supplied to fit and used for CG.

        Args:
            dataset: A Dataset object.
            max_rank (int): The maximum rank for the preconditioner, which
                uses a low-rank approximation to the matrix inverse. Larger
                numbers mean a more accurate approximation and thus reduce
                the number of iterations, but make the preconditioner more
                expensive to construct.
            method (str): one of "srht", "srht_2". "srht_2" runs two passes
                over the dataset. For the same max_rank, the preconditioner
                built by "srht_2" will reduce the number of CG iterations by
                25-30% compared with "srht", but it does incur the expense
                of a second pass over the dataset.

        Returns:
            preconditioner: A preconditioner object.
            achieved_ratio (float): The min eigval of the preconditioner over
                lambda, the noise hyperparameter shared between all kernels.
                This value has decent predictive value for assessing how
                well the preconditioner is likely to perform.
        """
        self._run_pre_fitting_prep(dataset, max_rank)
        x_mean = self._get_x_mean(dataset)
        if self.device == "gpu":
            preconditioner = Cuda_RandNysPreconditioner(self.kernel, dataset, max_rank,
                        self.verbose, self.random_seed, method,
                        x_mean = x_mean)
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
        else:
            preconditioner = CPU_RandNysPreconditioner(self.kernel, dataset, max_rank,
                        self.verbose, self.random_seed, method,
                        x_mean = x_mean)
        self._run_post_fitting_cleanup(dataset)
        return preconditioner, preconditioner.achieved_ratio



    def fit(self, dataset, preconditioner = None,
            tol:float = 1e-6,
            max_iter:int = 500,
            max_rank:int = 3000,
            min_rank:int = 512,
            mode:str = "cg",
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
            tol (float): The threshold below which iterative strategies (L-BFGS, CG,
                SGD) are deemed to have converged. Defaults to 1e-5. Note that how
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
            mode (str): Must be one of "cg", "lbfgs", "exact".
                Determines the approach used. If 'exact', self.kernel.get_num_rffs
                must be <= constants.constants.MAX_CLOSED_FORM_RFFS.
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
            losses (list): The loss on each iteration. Only for SGD and CG, otherwise,
                empty list.

        Raises:
            ValueError: The input dataset is checked for validity before tuning is
                initiated, an error is raised if problems are found."""
        self._run_pre_fitting_prep(dataset)
        self.weights = None
        self.n_classes = dataset.get_n_classes()

        if self.verbose:
            print("starting fitting")

        x_mean, targets = self._get_targets(dataset)

        if mode == "exact":
            n_iter = 1
            if self.kernel.get_num_rffs() > constants.MAX_CLOSED_FORM_RFFS:
                raise ValueError("You specified 'exact' fitting, but the number of rffs is "
                        f"> {constants.MAX_CLOSED_FORM_RFFS}.")
            self.weights = calc_discriminant_weights_exact(dataset, self.kernel, x_mean, targets)
        elif mode == "cg":
            if preconditioner is None:
                preconditioner = self._autoselect_preconditioner(dataset,
                        min_rank = min_rank, max_rank = max_rank,
                        ratio_target = autoselect_target_ratio,
                        always_use_srht2 = always_use_srht2,
                        x_mean = x_mean)
            self.weights, n_iter, _ = cg_fit_lib_discriminant(self.kernel, dataset,
                    x_mean, targets, tol, max_iter, preconditioner, self.verbose)
            if self.verbose:
                print(f"{n_iter} iterations.")

        elif mode == "lbfgs":
            model_fitter = lBFGSModelFit(dataset, self.kernel,
                    self.device, self.verbose, task_type = "discriminant",
                    n_classes = self.n_classes)
            self.weights, n_iter, _ = model_fitter.fit_model_lbfgs(max_iter, tol,
                    x_mean, targets)

        else:
            raise ValueError("Unrecognized fitting mode supplied. Must provide one of "
                        "'lbfgs', 'cg', 'exact'.")

        self._gamma = np.log(1 / self.n_classes) - 0.5 * (targets * self.weights).sum(axis=0)

        if self.verbose:
            print("Fitting complete.")
        if self.device == "gpu":
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
        self._run_post_fitting_cleanup(dataset)

        if run_diagnostics:
            return n_iter
