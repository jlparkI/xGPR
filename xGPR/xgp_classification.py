"""Describes the xGPDiscriminant class.

The xGPDiscriminant class provides the tools needed to fit a classification
model and make predictions for new datapoints. It inherits from
ModelBaseclass.
"""
import numpy as np
from scipy.optimize import minimize

try:
    import cupy as cp
    from cupyx.scipy.special import logsumexp as cupy_logsumexp
    from xGPR.xgpr_cuda_rfgen_cpp_ext import cudaFindClassMeans
    from .fitting_toolkit.cg_tools import GPU_ConjugateGrad
except:
    print("CuPy not detected. xGPR will run in CPU-only mode.")

from xGPR.xgpr_cpu_rfgen_cpp_ext import cpuFindClassMeans

from .fitting_toolkit.cg_tools import CPU_ConjugateGrad
from .fitting_toolkit.lsr1_toolkit import lSR1_classification
from .fitting_toolkit.nonlinear_cg_toolkit import nonlinear_CG_classification
from .fitting_toolkit.lbfgs_toolkit import lbfgs_cost_fun
from .constants import constants
from .model_baseclass import ModelBaseclass

from .fitting_toolkit.exact_fitting_toolkit import calc_discriminant_weights_exact
from .preconditioners.rand_nys_preconditioners import RandNysPreconditioner




class xGPDiscriminant(ModelBaseclass):
    """An approximate kernelized discriminant for classification.
    """

    def __init__(self, num_rffs:int = 256,
            kernel_choice:str = "RBF",
            device:str = "cpu",
            kernel_settings:dict = constants.DEFAULT_KERNEL_SPEC_PARMS,
            verbose:bool = True,
            uniform_priors = False,
            random_seed:int = 123,
            model_type:str = "discriminant"):
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
            uniform_priors (bool): If True, the prior probability of each class
                is assumed to be the same. If False (default), the prior probability
                of each class is determined by how frequently that class appears
                in the training data. This is a classification-specific argument.
            random_seed (int): The seed to the random number generator.
            model_type (str): One of "discriminant", "logistic". If "discriminant",
                fitting must be either via CG or exact methods. If "logistic",
                the model will be fitted using L-BFGS. Fitting logistic is slower
                because the fast preconditioned CG algorithm for fitting cannot
                be used but will sometimes be a little more accurate.
        """
        # Note that we pass 0 as the second argument for variance rffs
        # since the classifier does not currently compute variance.
        # Also notice that we set fit_intercept to False by default since
        # the classifier fits an intercept separately. This is a hack --
        # the fit_intercept argument is a little clunky anyway and
        # should probably be removed altogether.
        if not isinstance(kernel_settings, dict):
            raise RuntimeError("kernel_settings must be a dict.")
        if model_type == "discriminant":
            kernel_settings["intercept"] = False
        elif model_type == "logistic":
            kernel_settings["intercept"] = True
        else:
            raise RuntimeError("Unrecognized model type passed.")
        super().__init__(num_rffs, 0,
                        kernel_choice, device = device,
                        kernel_settings = kernel_settings,
                        verbose = verbose, random_seed = random_seed)

        self._uniform_priors = uniform_priors
        self._model_type = model_type


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


    def _get_class_means_priors(self, dataset):
        """Compiles the class-specific means, the weight for
        each class (the square root of the inverse of the
        number of instances) and the prior for each class
        (the number of instances divided by the total num
        training datapoints).

        Args:
            dataset: A valid Dataset object.

        Returns:
            class_means (ndarray): The mean of the random features across
                the whole dataset. Shape (num_rffs).
            class_weights (ndarray): The square root of the inverse of
                the number of datapoints in each class. Used for class
                weighting.
            priors (ndarray): The prior probability of each class, as
                determined from the training data.
        """
        self.n_classes = int(dataset.get_n_classes())

        if self.device == "cuda":
            class_means = cp.zeros((self.n_classes, self.kernel.get_num_rffs()))
            n_pts_per_class = cp.zeros((self.n_classes), dtype=cp.int64)

            for (xdata, ydata, ldata) in dataset.get_chunked_data():
                xfeatures, yclasses = self.kernel.transform_x_y(xdata, ydata, ldata,
                        classification=True)
                if ydata.max() > self.n_classes or ydata.min() < 0:
                    raise RuntimeError("Unexpected y-values encountered.")
                cudaFindClassMeans(xfeatures, class_means, yclasses, n_pts_per_class)

            if self._uniform_priors:
                log_priors = cp.zeros((n_pts_per_class.shape[0]))
            else:
                log_priors = cp.log((n_pts_per_class /
                    float(dataset.get_ndatapoints())).clip(min=1e-10))

            class_weights = cp.full(n_pts_per_class.shape[0],
                    (1./float(dataset.get_ndatapoints()))**0.5)

        else:
            class_means = np.zeros((self.n_classes, self.kernel.get_num_rffs()))
            n_pts_per_class = np.zeros((self.n_classes), dtype=np.int64)

            for (xdata, ydata, ldata) in dataset.get_chunked_data():
                xfeatures, yclasses = self.kernel.transform_x_y(xdata, ydata, ldata,
                        classification=True)
                if ydata.max() > self.n_classes or ydata.min() < 0:
                    raise RuntimeError("Unexpected y-values encountered.")
                cpuFindClassMeans(xfeatures, class_means, yclasses, n_pts_per_class)

            if self._uniform_priors:
                log_priors = np.zeros((n_pts_per_class.shape[0]))
            else:
                log_priors = np.log((n_pts_per_class /
                    float(dataset.get_ndatapoints())).clip(min=1e-10))

            class_weights = np.full(n_pts_per_class.shape[0],
                    (1./float(dataset.get_ndatapoints()))**0.5)

        class_means /= n_pts_per_class[:,None]
        return class_means, class_weights, log_priors



    def build_preconditioner(self, dataset, max_rank:int = 512,
            method:str = "srht"):
        """Builds a preconditioner. The resulting preconditioner object
        can be supplied to fit and used for CG. Use this function if you do
        not want fit() to automatically choose preconditioner settings for
        you.

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
        # Only difference between regression and classification is this
        # feature which is nonetheless crucial since the classifier must
        # find the class means, priors and class weights.
        class_means, class_weights, _ = self._get_class_means_priors(dataset)

        preconditioner = RandNysPreconditioner(self.kernel, dataset, max_rank,
                        self.verbose, self.random_seed, method,
                        class_means=class_means, class_weights=class_weights)

        if self.device == "cuda":
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
        return preconditioner, preconditioner.achieved_ratio


    def fit(self, dataset, preconditioner = None,
            tol:float = 1e-3,
            history_size:int = 5,
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
            tol (float): The threshold below which iterative strategies (CG)
                are deemed to have converged. Defaults to 1e-5. Note that how
                reaching the threshold is assessed may depend on the algorithm.
            history_size (int): The maximum history size to store for L-SR1
                fitting. The default is usually fine as long as a preconditioner
                is being used.
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
            mode (str): Must be one of "cg", "exact", "lsr1".
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

        if mode not in ("lsr1", "cg", "exact"):
            raise RuntimeError("Unrecognized fitting mode supplied. Must provide one of "
                        "'lsr1', 'cg', 'exact'.")

        if self.verbose:
            print("starting fitting")

        class_means, class_weights, priors = self._get_class_means_priors(dataset)
        if self.device == "cuda":
            norm_constant = float(cp.linalg.norm(class_means, axis=1).max())
        else:
            norm_constant = float(np.linalg.norm(class_means, axis=1).max())

        if mode == "exact":
            if self._model_type != "discriminant":
                raise RuntimeError("This fitting mode is only allowed for discriminant models. "
                        "To use this, set model type to 'discriminant'")

            if self.device == "cuda":
                targets = cp.ascontiguousarray(class_means.T)
            else:
                targets = np.ascontiguousarray(class_means.T)

            targets /= norm_constant

            n_iter = 1
            if self.kernel.get_num_rffs() > constants.MAX_CLOSED_FORM_RFFS:
                raise RuntimeError("You specified 'exact' fitting, but the number of rffs is "
                        f"> {constants.MAX_CLOSED_FORM_RFFS}.")
            self.weights = calc_discriminant_weights_exact(dataset, self.kernel,
                    targets, class_means, class_weights)
            losses = []
            self.gamma = (priors / norm_constant) - 0.5 * norm_constant * \
                    (class_means.T * self.weights).sum(axis=0)

        # Importantly, note that if using lbfgs, we fit using CG FIRST,
        # which is for LDA, then shift to logistic regression and use
        # the LDA result as a starting point for optimization for a
        # logistic regression objective with L-BFGS as the optimization
        # algorithm.
        elif mode == "cg":
            if preconditioner is None:
                preconditioner = self._autoselect_preconditioner(dataset,
                        min_rank = min_rank, max_rank = max_rank,
                        ratio_target = autoselect_target_ratio,
                        always_use_srht2 = always_use_srht2)
            cg_operator = nonlinear_CG_classification(dataset, self.kernel,
                    self.device, self.verbose, preconditioner,
                    history_size=history_size)

            self.weights, n_iter, losses = cg_operator.fit_model(max_iter, tol)
            if self.device == "cuda":
                self.gamma = cp.zeros((self.n_classes))
            else:
                self.gamma = np.zeros((self.n_classes))
            if self.verbose:
                print(f"CG iterations: {n_iter}")


        if mode == "lsr1":
            if preconditioner is None:
                preconditioner = self._autoselect_preconditioner(dataset,
                        min_rank = min_rank, max_rank = max_rank,
                        ratio_target = autoselect_target_ratio,
                        always_use_srht2 = always_use_srht2)
            lsr1_operator = lSR1_classification(dataset, self.kernel,
                    self.device, self.verbose, preconditioner,
                    history_size=history_size)

            self.weights, n_iter, losses = lsr1_operator.fit_model(max_iter, tol)
            if self.device == "cuda":
                self.gamma = cp.zeros((self.n_classes))
            else:
                self.gamma = np.zeros((self.n_classes))
            if self.verbose:
                print(f"LSR1 iterations: {n_iter}")



        '''if mode == "lbfgs":
            if preconditioner is None:
                preconditioner = self._autoselect_preconditioner(dataset,
                        min_rank = min_rank, max_rank = max_rank,
                        ratio_target = autoselect_target_ratio,
                        always_use_srht2 = always_use_srht2)
            losses = []
            x0 = np.zeros((self.num_rffs * dataset.get_n_classes() ))
            if self.device == "cuda":
                self.gamma = cp.zeros((self.n_classes))
            else:
                self.gamma = np.zeros((self.n_classes))

            weights, _, info_dict = xgpr_lbfgs(func=lbfgs_cost_fun, x0=x0,
                    maxfun=max_iter,
                    args=(dataset, self.kernel, class_means.shape[0],
                        self.gamma),
                    num_rffs=self.num_rffs, n_classes=dataset.get_n_classes(),
                    preconditioner=preconditioner)

            if self.device == "cuda":
                self.weights = cp.zeros((self.num_rffs, class_means.shape[0]))
                for k, i in enumerate(range(0, weights.shape[0], self.num_rffs)):
                    self.weights[:,k] = cp.asarray(weights.x[i:i+self.num_rffs])
            else:
                self.weights = np.zeros((self.num_rffs, class_means.shape[0]))
                for k, i in enumerate(range(0, weights.x.shape[0], self.num_rffs)):
                    self.weights[:,k] = weights.x[i:i+self.num_rffs]

            n_iter = info_dict['funcalls']'''



        if self.verbose:
            print("Fitting complete.")
        if self.device == "cuda":
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()

        if run_diagnostics:
            return n_iter, losses
