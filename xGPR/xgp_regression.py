"""Describes the xGPRegression class.

The xGPRegression class provides the tools needed to fit a regression
model and make predictions for new datapoints. It inherits from
ModelBaseclass.
"""
import warnings
import numpy as np
from scipy.optimize import minimize
from .cg_toolkit.cg_tools import CPU_ConjugateGrad

try:
    import cupy as cp
    from .preconditioners.cuda_rand_nys_preconditioners import Cuda_RandNysPreconditioner
    from .cg_toolkit.cg_tools import GPU_ConjugateGrad
except:
    print("CuPy not detected. xGPR will run in CPU-only mode.")

from .constants import constants
from .model_baseclass import ModelBaseclass
from .preconditioners.tuning_preconditioners import RandNysTuningPreconditioner
from .preconditioners.inter_device_preconditioners import InterDevicePreconditioner
from .preconditioners.rand_nys_preconditioners import CPU_RandNysPreconditioner

from .fitting_toolkit.cg_fitting_toolkit import cg_fit_lib_internal
from .fitting_toolkit.exact_fitting_toolkit import calc_weights_exact, calc_variance_exact

from .scoring_toolkit.approximate_nmll_calcs import estimate_logdet
from .scoring_toolkit.nmll_gradient_tools import exact_nmll_reg_grad, calc_gradient_terms
from .scoring_toolkit.probe_generators import generate_normal_probes_gpu
from .scoring_toolkit.probe_generators import generate_normal_probes_cpu
from .scoring_toolkit.exact_nmll_calcs import calc_design_mat, direct_weight_calc
from .scoring_toolkit.bayes_grid import bayes_grid_tuning
from .scoring_toolkit.lb_optimizer import shared_hparam_search
from .scoring_toolkit.alpha_beta_optimizer import optimize_alpha_beta




class xGPRegression(ModelBaseclass):
    """An approximate Gaussian process for regression.
    """

    def __init__(self, num_rffs:int = 256,
                    variance_rffs:int = 16,
                    kernel_choice:str = "RBF",
                    device:str = "cpu",
                    kernel_settings:dict = constants.DEFAULT_KERNEL_SPEC_PARMS,
                    verbose:bool = True,
                    num_threads:int = 2,
                    random_seed:int = 123) -> None:
        """The constructor for xGPRegression. Passes arguments onto
        the parent class constructor.

        Args:
            num_rffs (int): The number of random Fourier features
                to use.
            variance_rffs (int): The number of random Fourier features
                to use for posterior predictive variance (i.e. calculating
                uncertainty on predictions). Defaults to 64.
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
            random_seed (int): The seed to the random number generator.
        """
        super().__init__(num_rffs, variance_rffs,
                        kernel_choice, device = device,
                        kernel_settings = kernel_settings,
                        verbose = verbose, num_threads = num_threads,
                        random_seed = random_seed)



    def predict(self, input_x, sequence_lengths = None, get_var:bool = False,
            chunk_size:int = 2000):
        """Generate a predicted value for each
        input datapoint -- and if desired the variance.

        Args:
            input_x (np.ndarray): The input data. Should be a 2d numpy
                array (if non-convolution kernel) or 3d (if convolution
                kernel).
            sequence_lengths: None if you are using a fixed-vector kernel (e.g.
                RBF) and a 1d array of the number of elements in each sequence /
                nodes in each graph if you are using a graph or Conv1d kernel.
            get_var (bool): If True, return (predictions, variance).
                If False, return predictions.
            chunk_size (int): The number of datapoints to process at
                a time. Lower values limit memory consumption. Defaults
                to 2000.

        Returns:
            If get_var is True, returns (predictions, variance). If False,
            returns predictions. Both are numpy arrays of length N
            for N datapoints.

        Raises:
            ValueError: If the dimesionality or type of the input does
                not match what is expected, or if the model has
                not yet been fitted, a ValueError is raised.
        """
        self.pre_prediction_checks(input_x, sequence_lengths, get_var)
        preds, var = [], []

        lambda_ = self.kernel.get_lambda()

        for i in range(0, input_x.shape[0], chunk_size):
            cutoff = min(i + chunk_size, input_x.shape[0])
            if sequence_lengths is not None:
                xfeatures = self.kernel.transform_x(input_x[i:cutoff,...],
                        sequence_lengths[i:cutoff])
            else:
                xfeatures = self.kernel.transform_x(input_x[i:cutoff,...])

            preds.append((xfeatures * self.weights[None, :]).sum(axis = 1))

            if get_var:
                if self.exact_var_calculation:
                    xfeatures = xfeatures[:,:self.variance_rffs]
                    pred_var = (self.var @ xfeatures.T).T
                else:
                    pred_var = self.var.batch_matvec(xfeatures.T).T
                pred_var = lambda_**2 + lambda_**2 * (xfeatures * pred_var).sum(axis=1)
                var.append(pred_var)

        if self.device == "cuda":
            preds = cp.asnumpy(cp.concatenate(preds))
        else:
            preds = np.concatenate(preds)
        if not get_var:
            return preds * self.trainy_std + self.trainy_mean
        if self.device == "cuda":
            var = cp.asnumpy(cp.concatenate(var))
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
        else:
            var = np.concatenate(var)
        #There may (rarely) be numerical issues which result in variances
        #less than zero; if so, set those variances to zero.
        var[var < 0] = 0
        return preds * self.trainy_std + self.trainy_mean, var * self.trainy_std**2


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
        if self.device == "cuda":
            preconditioner = Cuda_RandNysPreconditioner(self.kernel, dataset, max_rank,
                        self.verbose, self.random_seed, method)
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
        else:
            preconditioner = CPU_RandNysPreconditioner(self.kernel, dataset, max_rank,
                        self.verbose, self.random_seed, method)
        return preconditioner, preconditioner.achieved_ratio




    def exact_nmll(self, hyperparams, dataset):
        """Calculates the exact negative marginal log likelihood (the model
        'score') using matrix decompositions. Fast for small numbers of random
        features but poor scaling to larger numbers. Will return a default
        large number if numerical instability is encountered (only occurs
        for extreme hyperparameter values).

        Args:
            hyperparams (np.ndarray): A numpy array containing the new
                set of hyperparameters that should be assigned to the kernel.
            dataset: An OnlineDataset or OfflineDataset containing the raw
                data we will use for this evaluation.

        Returns:
            negloglik (float): The negative marginal log likelihood for the
                input hyperparameters.
        """
        self._run_singlepoint_nmll_prep(dataset, exact_method = True)

        self.kernel.set_hyperparams(hyperparams, logspace=True)
        ndatapoints = dataset.get_ndatapoints()
        z_trans_z, z_trans_y, y_trans_y = calc_design_mat(dataset, self.kernel)
        if self.device == "cpu":
            diagfunc, logfunc = np.diag, np.log
        else:
            diagfunc, logfunc = cp.diag, cp.log

        #Direct weight calculation may fail IF the hyperparameters supplied
        #lead to a singular design matrix. This is rare, but be prepared to
        #handle if this problem is encountered.
        try:
            chol_z_trans_z, weights = direct_weight_calc(z_trans_z, z_trans_y,
                    self.kernel)
        except:
            warnings.warn("Near-singular matrix encountered when calculating score for "
                    f"hyperparameters {hyperparams}.")
            return constants.DEFAULT_SCORE_IF_PROBLEM

        nll1 = float(0.5 * (y_trans_y - z_trans_y.T @ weights))
        nll2 = float(logfunc(diagfunc(chol_z_trans_z)).sum())
        negloglik, _ = optimize_alpha_beta(self.kernel.get_lambda(),
                np.array([nll1, nll2]), ndatapoints, self.kernel.get_num_rffs())

        #Direct weight calculation may fail IF the hyperparameters supplied
        #lead to a singular design matrix. This is rare, but be prepared to
        #handle if this problem is encountered.
        if np.isnan(negloglik):
            warnings.warn("Near-singular matrix encountered when calculating score for "
                    f"hyperparameters {hyperparams}.")
            return constants.DEFAULT_SCORE_IF_PROBLEM

        if self.verbose:
            print("Evaluated NMLL.")
        return negloglik



    def exact_nmll_gradient(self, hyperparams, dataset, subsample:float = 1):
        """Calculates the gradient of the negative marginal log likelihood w/r/t
        the hyperparameters for a specified set of hyperparameters using
        exact methods (matrix decompositions). Fast for small numbers of
        random features but poor scaling to larger numbers. Will return a default
        large number if numerical instability is encountered (only occurs
        for extreme hyperparameter values).

        Args:
            hyperparams (np.ndarray): The set of hyperparameters at which
                to calculate the gradient.
            dataset: An Online or OfflineDataset containing the raw data
                we will use for this evaluation.
            subsample (float): A value in the range [0.1,1] that indicates what
                fraction of the training set to use each time the gradient is
                calculated (the same subset is used every time). In general, 1
                will give better results, but using a subsampled subset can be
                a fast way to find the (approximate) location of a good
                hyperparameter set.

        Returns:
            negloglik (float): The negative marginal log likelihood.
            grad (np.ndarray): The gradient of the NMLL w/r/t the hyperparameters.
        """
        self._run_singlepoint_nmll_prep(dataset, exact_method = True)

        init_hparams = self.kernel.get_hyperparams()
        self.kernel.set_hyperparams(hyperparams, logspace=True)
        hparams = self.kernel.get_hyperparams(logspace=False)

        if self.verbose:
            print("Evaluating gradient...")

        z_trans_z, z_trans_y, y_trans_y, dz_dsigma_ty, inner_deriv, nsamples = \
                        calc_gradient_terms(dataset, self.kernel, self.device, subsample)

        #Try-except here since very occasionally, optimizer samples a really
        #terrible set of hyperparameters that with numerical error has resulted in a
        #non-positive-definite design matrix. TODO: Find a good workaround to
        #avoid this whole problem.
        try:
            negloglik, grad, _ = exact_nmll_reg_grad(z_trans_z, z_trans_y,
                        y_trans_y, hparams, nsamples, dz_dsigma_ty,
                        inner_deriv, self.device)
        except Exception as _:
            return constants.DEFAULT_SCORE_IF_PROBLEM, hyperparams - init_hparams

        #Same problem as above, only occasionally the LAPACK routines return
        #nan instead of raising an error.
        if np.isnan(negloglik):
            return constants.DEFAULT_SCORE_IF_PROBLEM, hyperparams - init_hparams
        return float(negloglik), grad



    def approximate_nmll(self, hyperparams, dataset, manual_settings:dict = None):
        """Calculates the approximate negative marginal log likelihood (the model
        'score') using stochastic Lanczos quadrature with preconditioning.
        Slower than exact for very small numbers of random features, but
        much faster for large (much better scaling). Avoids numerical
        stability issues, but as an approximation may contain some error,
        especially if fitting parameters are chosen poorly.

        Args:
            hyperparams (np.ndarray): A numpy array containing the new
                set of hyperparameters that should be assigned to the kernel.
            dataset: An OnlineDataset or OfflineDataset containing the raw
                data we will use for this evaluation.
            manual_settings: Either None (default) or a dict. If None, the function will
                try to automatically choose settings that will give a good
                approximation. This process usually works well but you may
                occasionally be able to achieve a better approximation and / or
                better speed by choosing settings yourself. If manual_settings is
                a dict, it can contain the following settings:
                
                * ``"max_rank"``: The preconditioner rank for approximate NMLL estimation.
                  A larger value may improve estimation accuracy, but at the
                  expense of speed. 512 - 1024 is fine for noisy data, 2048 is
                  better if data is close to noise free.

                * ``"nsamples"``: The number of probes for approximate NMLL estimation. 25 (default)
                  is usually fine.

                * ``"niter"``: The maximum number of iterations for approximate NMLL. Should
                  only really become an issue if CG is failing to fit in under 500 iter (the
                  default) -- unusual, hyperparameters that give rise to this issue are likely
                  poor in most cases anyway.

                * ``"tol"``: The convergence tolerance for approximate NMLL. 1e-6 (default)
                  is usually fine. A tighter tol (e.g. 1e-7) can improve accuracy slightly
                  at the expense of increased cost.

                * ``"preconditioner_mode"``: One of "srht", "srht_2". Determines the mode of
                  preconditioner construction.

        Returns:
            nmll (float): The negative marginal log likelihood for the
                input hyperparameters.
        """
        self._run_singlepoint_nmll_prep(dataset, exact_method = False)

        self.kernel.set_hyperparams(hyperparams, logspace = True)
        if self.verbose:
            print("Now building preconditioner...")

        #If the user supplied manual settings, they want to control preconditioner
        #construction. Build a preconditioner using the default settings modified
        #with any settings they selected.
        settings = constants.default_nmll_params
        if manual_settings is not None:
            for key in settings:
                if key in manual_settings:
                    settings[key] = manual_settings[key]

            if settings["max_rank"] >= self.num_rffs:
                settings["max_rank"] = self.num_rffs - 1

            preconditioner = RandNysTuningPreconditioner(self.kernel, dataset,
                        settings["max_rank"], False, self.random_seed,
                        settings["preconditioner_mode"])
        else:
            preconditioner = self._autoselect_preconditioner(dataset,
                    min_rank = constants.SMALLEST_NMLL_MAX_RANK,
                    max_rank = constants.LARGEST_NMLL_MAX_RANK,
                    always_use_srht2 = True,
                    tuning = True)

        if self.verbose:
            print("Now fitting...")

        if self.device == "cuda":
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
            cg_operator = GPU_ConjugateGrad()
            resid = cp.zeros((self.kernel.get_num_rffs(), 2, settings["nsamples"] + 1),
                            dtype = cp.float64)
            probes = generate_normal_probes_gpu(settings["nsamples"],
                    self.kernel.get_num_rffs(), self.random_seed, preconditioner)
        else:
            cg_operator = CPU_ConjugateGrad()
            resid = np.zeros((self.kernel.get_num_rffs(), 2,
                            settings["nsamples"] + 1), dtype = np.float64)
            probes = generate_normal_probes_cpu(settings["nsamples"],
                self.kernel.get_num_rffs(), self.random_seed, preconditioner)

        z_trans_y = preconditioner.get_zty()
        y_trans_y = preconditioner.get_yty()

        resid[:,0,0] = z_trans_y / dataset.get_ndatapoints()
        resid[:,0,1:] = probes

        x_k, alphas, betas = cg_operator.fit(dataset, self.kernel,
                    preconditioner, resid, settings["nmll_iter"],
                    settings["nmll_tol"], verbose = False,
                    nmll_settings = True)

        x_k[:,0] *= dataset.get_ndatapoints()

        logdet = estimate_logdet(alphas, betas, self.kernel.get_num_rffs(),
                        preconditioner, self.device)

        nll1 = float(0.5 * (y_trans_y - z_trans_y.T @ x_k[:,0]))

        negloglik, _ = optimize_alpha_beta(self.kernel.get_lambda(),
                np.array([nll1, 0.5 * logdet]), dataset.get_ndatapoints(),
                self.kernel.get_num_rffs())
        if self.verbose:
            print("NMLL evaluation completed.")

        return negloglik



    def fit(self, dataset, preconditioner = None,
            tol:float = 1e-6, max_iter:int = 500,
            mode:str = "cg", suppress_var:bool = False,
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
            tol (float): The threshold below which iterative strategies (L-BFGS, CG)
                are deemed to have converged.
            max_iter (int): The maximum number of epochs for iterative strategies.
            mode (str): Must be one of "cg", "exact".
                Determines the approach used. If 'exact', self.kernel.get_num_rffs
                must be <= constants.constants.MAX_CLOSED_FORM_RFFS.
            suppress_var (bool): If True, do not calculate variance. Use this when you
                are just testing performance on a validation set and so do not need to
                calculate variance.
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
            losses (list): The loss on each iteration (only for mode='cg'; empty list
                otherwise).

        Raises:
            ValueError: The input dataset is checked for validity before tuning is
                initiated, an error is raised if problems are found."""
        self._run_pre_fitting_prep(dataset)
        self.weights, self.var = None, None
        self.exact_var_calculation = True

        if self.verbose:
            print("starting fitting")

        if mode == "exact":
            if self.kernel.get_num_rffs() > constants.MAX_CLOSED_FORM_RFFS:
                raise ValueError("You specified 'exact' fitting, but the number of rffs is "
                        f"> {constants.MAX_CLOSED_FORM_RFFS}.")
            self.weights, n_iter, losses = calc_weights_exact(dataset, self.kernel)

        elif mode == "cg":
            if preconditioner is None:
                preconditioner = self._autoselect_preconditioner(dataset,
                        min_rank = min_rank, max_rank = max_rank,
                        ratio_target = autoselect_target_ratio,
                        always_use_srht2 = always_use_srht2)
            self.weights, n_iter, losses = cg_fit_lib_internal(self.kernel, dataset, tol,
                    max_iter, preconditioner, self.verbose)

        else:
            raise ValueError("Unrecognized fitting mode supplied. Must provide one of "
                        "'cg', 'exact'.")

        if not suppress_var:
            if self.verbose:
                print("Now performing variance calculations...")
            #We use exact variance calc UNLESS we are dealing with a linear kernel with
            #a very large number of input features. Find a better / more satisfactory
            #way to resolve this...
            if "Linear" in self.kernel_choice:
                self.var = InterDevicePreconditioner(self.kernel, dataset,
                        self.variance_rffs, False, self.random_seed, "srht")
                self.exact_var_calculation = False
            else:
                self.var = calc_variance_exact(self.kernel, dataset, self.kernel_choice,
                                self.variance_rffs)


        if self.verbose:
            print("Fitting complete.")
        if self.device == "cuda":
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()

        if run_diagnostics:
            return n_iter, losses



    #############################
    #The next block of functions are used for tuning hyperparameters. We provide
    #several strategies and of course the user can easily create their own
    #(e.g. using Optuna).
    #############################


    def tune_hyperparams_crude(self, dataset, bounds = None,
                    random_seed:int = 123,
                    max_bayes_iter:int = 30,
                    subsample:float = 1):
        """Tunes the hyperparameters using Bayesian optimization, but with
        a 'trick' that simplifies the problem greatly for 1-3 hyperparameter
        kernels. Hyperparameters are scored using an exact NMLL calculation.
        The NMLL calculation uses a matrix decomposition with cubic
        scaling in the number of random features, so it is slow
        for anything more than 3-4,000 random features, but has
        low risk of overfitting and is easy to use. It is therefore
        intended as a "quick-and-dirty" method. We recommend using
        this with a small number of random features (e.g. 500 - 3000)
        and call self.tune_hyperparams to fine-tune further if needed,
        or fine-tune using an outside library (Optuna) or even simple
        grid search.

        Args:
            dataset: Object of class OnlineDataset or OfflineDataset.
            random_seed (int): A random seed for the random
                number generator. Defaults to 123.
            max_bayes_iter (int): The maximum number of iterations of Bayesian
                optimization.
            bounds (np.ndarray): The bounds for optimization. If None, default
                boundaries for the kernel will be used. Otherwise, must be an
                array of shape (# hyperparams, 2).
            subsample (float): A value in the range [0.01,1] that indicates what
                fraction of the training set to use each time the score is
                calculated (the same subset is used every time). In general, 1
                will give better results, but using a subsampled subset can be
                a fast way to find the (approximate) location of a good
                hyperparameter set.

        Returns:
            hyperparams (np.ndarray): The best hyperparams found during optimization.
            n_feval (int): The number of function evaluations during optimization.
            best_score (float): The best negative marginal log-likelihood achieved.

        Raises:
            ValueError: The input dataset is checked for validity before tuning is
                initiated. If problems are found, a raise exception will provide an
                explanation of the error. This method will also raise an exception
                if you try to use it on a kernel with > 4 hyperparameters (since
                this strategy no longer provides any benefit under those conditions).
        """
        if subsample < 0.01 or subsample > 1:
            raise ValueError("subsample must be in the range [0.01, 1].")

        optim_bounds = self._run_pre_nmll_prep(dataset, bounds)
        num_hparams = self.kernel.get_hyperparams().shape[0]
        if num_hparams == 1:
            best_score, hyperparams = shared_hparam_search(np.array([]), self.kernel,
                    dataset, optim_bounds, subsample = subsample)
            n_feval = 1
        elif 4 > num_hparams > 1:
            hyperparams, _, best_score, n_feval = bayes_grid_tuning(self.kernel,
                                dataset, optim_bounds, random_seed, max_bayes_iter,
                                self.verbose, subsample = subsample)

        else:
            raise ValueError("The crude procedure is only appropriate for "
                    "kernels with 1-3 hyperparameters.")

        self.kernel.set_hyperparams(hyperparams, logspace=True)
        return hyperparams, n_feval, best_score


    def tune_hyperparams(self, dataset, bounds = None,
            max_iter:int = 50, tuning_method:str = "Powell",
            starting_hyperparams = None, tol:float = 1e-2,
            n_restarts:int = 1, nmll_method:str = "exact",
            manual_settings:dict = None):
        """Tunes the hyperparameters WITHOUT using gradient information.
        The methods included here can be extremely efficient for one-
        and two-hyperparameter kernels but are not recommended for kernels
        with many hyperparameters. This method can either use exact
        NMLL (small num_rffs) or approximate (scales to larger num_rffs, but
        approximation is only good so long as settings selected are reasonable).

        Args:
            dataset: Object of class OnlineDataset or OfflineDataset.
            bounds (np.ndarray): The bounds for optimization. If None, default
                boundaries for the kernel will be used. Otherwise, must be an
                array of shape (# hyperparams, 2).
            max_iter (int): The maximum number of iterations.
            tuning_method (str): One of 'Powell', 'Nelder-Mead', 'L-BFGS-B'.
                'Nelder-Mead' and 'Powell' do not perform gradient calculations
                so have much better scaling to large numbers of RFFs but may take
                more iterations to converge. 'Nelder-Mead' takes longer
                to converge than 'Powell' but is more precise.
                'L-BFGS-B' uses exact gradient calculations if 'nmll_method=exact',
                which have poor scaling to > 4000 RFFs but can converge quite
                quickly. With 'nmll_method=approximate', it uses a finite
                difference approximation. For a good guideline on which
                method to prefer, see the User Guide.
            starting_hyperparams: Either None or a numpy array of shape (nhparams)
                where (nhparams) is the number of hyperparameters for the selected
                kernel. If None, a default is used.
            tol (float): The largest shift in NMLL before the algorithm is
                asssumed to have converged.
            n_restarts (int): The number of times to restart the optimizer from
                a new random starting point.
            nmll_method (str): One of 'exact', 'approximate'. 'Exact' is very fast
                if num_rffs is small but has cubic scaling with num_rffs. On GPU,
                it can work well up to 8,000 RFFs or so.
                'approximate' is slow but has better scaling (i.e. the
                time taken increases linearly with the number of RFFs).
            manual_settings: Either None (default) or a dict. If None, the function will
                try to automatically choose settings that will give a good
                approximation. This process usually works well but you may
                occasionally be able to achieve a better approximation and / or
                better speed by choosing settings yourself. If manual_settings is
                a dict, it can contain the following settings:
                
                * ``"max_rank"``: The preconditioner rank for approximate NMLL estimation.
                  A larger value may improve estimation accuracy, but at the
                  expense of speed. 512 - 1024 is fine for noisy data, 2048 is
                  better if data is close to noise free.

                * ``"nsamples"``: The number of probes for approximate NMLL estimation. 25 (default) is usually fine.

                * ``"niter"``: The maximum number of iterations for approximate NMLL. Should
                  only really become an issue if CG is failing to fit in under 500 iter (the
                  default) -- unusual, hyperparameters that give rise to this issue are likely
                  poor in most cases anyway.

                * ``"tol"``: The convergence tolerance for approximate NMLL. 1e-6 (default)
                  is usually fine. A tighter tol (e.g. 1e-7) can improve accuracy slightly
                  at the expense of increased cost.

                * ``"preconditioner_mode"``: One of "srht", "srht_2". Determines the mode of
                  preconditioner construction.

        Returns:
            hyperparams (np.ndarray): The best hyperparams found during optimization.
            n_feval (int): The number of function evaluations during optimization.
            best_score (float): The best negative marginal log-likelihood achieved.

        Raises:
            ValueError: A ValueError is raised if invalid inputs are supplied.
        """
        if tuning_method == "Powell":
            options={"maxfev":max_iter, "xtol":1e-1, "ftol":tol}
        elif tuning_method == "Nelder-Mead":
            options={"maxfev":max_iter, "fatol":tol}
        elif tuning_method == "L-BFGS-B":
            if nmll_method == "approximate":
                raise ValueError("Approximate NMLL is not supported for "
                        "L-BFGS-B at this time.")
            options={"maxiter":max_iter, "ftol":tol}
        else:
            raise ValueError("Invalid tuning method supplied.")

        optim_bounds = self._run_pre_nmll_prep(dataset, bounds)

        if nmll_method == "approximate":
            args = (dataset, manual_settings)
            cost_fun = self.approximate_nmll
        elif nmll_method == "exact":
            args = (dataset,)
            if tuning_method == "L-BFGS-B":
                cost_fun = self.exact_nmll_gradient
            else:
                cost_fun = self.exact_nmll
        else:
            raise ValueError("Invalid nmll method supplied.")


        bounds_tuples = list(map(tuple, optim_bounds))
        rng = np.random.default_rng(self.random_seed)

        if starting_hyperparams is None:
            x0 = self.kernel.get_hyperparams()
            if (x0 - optim_bounds[:,0]).min() < 0 or \
                    (optim_bounds[:,1] - x0).min() < 0:
                x0 = optim_bounds.mean(axis=1)
                warnings.warn("The kernel hyperparameters were outside the "
                    "optimization boundaries. The mean of the optimization "
                    "boundaries will be used as a starting point.", UserWarning)

        elif isinstance(starting_hyperparams, np.ndarray) and \
            starting_hyperparams.shape[0] == self.kernel.get_hyperparams().shape[0]:
            x0 = starting_hyperparams
        else:
            raise ValueError("Invalid starting hyperparams were supplied.")

        best_score, n_feval = np.inf, 0

        for _ in range(n_restarts):
            if tuning_method != "L-BFGS-B":
                res = minimize(cost_fun, x0 = x0,
                        options=options, method=tuning_method,
                        args = args, bounds = bounds_tuples)
            else:
                res = minimize(cost_fun, x0 = x0,
                        options=options, method=tuning_method,
                        args = args, bounds = bounds_tuples,
                        jac = True)

            n_feval += res.nfev

            if res.fun < best_score:
                n_feval, hyperparams, best_score = res.nfev, res.x, res.fun
            if self.verbose:
                print(f"Best score: {best_score}")
            x0 = [rng.uniform(low = optim_bounds[j,0],
                    high = optim_bounds[j,1]) for j in range(optim_bounds.shape[0])]
            x0 = np.asarray(x0)

        self.kernel.set_hyperparams(hyperparams, logspace=True)
        return hyperparams, n_feval, best_score
