"""Describes the xGPRegression class.

The xGPRegression class provides the tools needed to fit a regression
model and make predictions for new datapoints. It inherits from
ModelBaseclass.
"""
import warnings
try:
    import cupy as cp
    from .preconditioners.cuda_rand_nys_preconditioners import Cuda_RandNysPreconditioner
except:
    print("CuPy not detected. xGPR will run in CPU-only mode.")
import numpy as np
from scipy.optimize import minimize

from .constants import constants
from .model_baseclass import ModelBaseclass
from .preconditioners.rand_nys_preconditioners import CPU_RandNysPreconditioner
from .preconditioners.tuning_preconditioners import RandNysTuningPreconditioner
from .preconditioners.inter_device_preconditioners import InterDevicePreconditioner

from .fitting_toolkit.lbfgs_fitting_toolkit import lBFGSModelFit
from .fitting_toolkit.cg_fitting_toolkit import cg_fit_lib_ext, cg_fit_lib_internal
from .fitting_toolkit.exact_fitting_toolkit import calc_weights_exact, calc_variance_exact

from .scoring_toolkit.approximate_nmll_calcs import estimate_logdet
from .scoring_toolkit.nmll_gradient_tools import exact_nmll_reg_grad, calc_gradient_terms
from .scoring_toolkit.probe_generators import generate_normal_probes_gpu
from .scoring_toolkit.probe_generators import generate_normal_probes_cpu
from .scoring_toolkit.exact_nmll_calcs import calc_zty, calc_design_mat, direct_weight_calc
from .scoring_toolkit.bayes_grid import bayes_grid_tuning
from .scoring_toolkit.lb_optimizer import shared_hparam_search
from .scoring_toolkit.alpha_beta_optimizer import optimize_alpha_beta

from cg_tools import CPU_ConjugateGrad, GPU_ConjugateGrad



class xGPRegression(ModelBaseclass):
    """A subclass of GPRegressionBaseclass that houses methods
    unique to regression problems. Only attributes not shared by
    the parent class are described here.
    """

    def __init__(self, num_rffs = 256, variance_rffs = 16,
                    kernel_choice="RBF",
                    device = "cpu",
                    kernel_specific_params = constants.DEFAULT_KERNEL_SPEC_PARMS,
                    verbose = True,
                    num_threads = 2,
                    random_seed = 123):
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
            kernel_specific_params (dict): Contains kernel-specific parameters --
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
                        kernel_specific_params = kernel_specific_params,
                        verbose = verbose, num_threads = num_threads,
                        random_seed = random_seed)



    def predict(self, input_x, get_var = False,
            chunk_size = 2000):
        """Generate a predicted value for each
        input datapoint -- and if desired the variance.

        Args:
            input_x (np.ndarray): The input data. Should be a 2d numpy
                array (if non-convolution kernel) or 3d (if convolution
                kernel).
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
        xdata = self.pre_prediction_checks(input_x, get_var)
        preds, var = [], []

        lambda_ = self.kernel.get_lambda()

        for i in range(0, xdata.shape[0], chunk_size):
            cutoff = min(i + chunk_size, xdata.shape[0])
            xfeatures = self.kernel.transform_x(xdata[i:cutoff, :])
            preds.append((xfeatures * self.weights[None, :]).sum(axis = 1))
            if get_var:
                if self.exact_var_calculation:
                    xfeatures = xfeatures[:,:self.variance_rffs]
                    pred_var = (self.var @ xfeatures.T).T
                else:
                    pred_var = self.var.batch_matvec(xfeatures.T).T
                pred_var = lambda_**2 + lambda_**2 * (xfeatures * pred_var).sum(axis=1)
                var.append(pred_var)

        if self.device == "gpu":
            preds = cp.asnumpy(cp.concatenate(preds))
        else:
            preds = np.concatenate(preds)
        if not get_var:
            return preds * self.trainy_std + self.trainy_mean
        if self.device == "gpu":
            var = cp.asnumpy(cp.concatenate(var))
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
        else:
            var = np.concatenate(var)
        #There may (rarely) be numerical issues which result in variances
        #less than zero; if so, set those variances to zero.
        var[var < 0] = 0
        return preds * self.trainy_std + self.trainy_mean, var * self.trainy_std**2




    def build_preconditioner(self, dataset, max_rank = 512,
                        preset_hyperparams = None, method = "srht"):
        """Builds a preconditioner. The resulting preconditioner object
        can be supplied to fit and used for CG, L-BFGS, SGD etc.

        Args:
            dataset: A Dataset object.
            max_rank (int): The maximum rank for the preconditioner, which
                uses a low-rank approximation to the matrix inverse. Larger
                numbers mean a more accurate approximation and thus reduce
                the number of iterations, but make the preconditioner more
                expensive to construct.
            preset_hyperparams: Either None or a numpy array. If None,
                hyperparameters must already have been tuned using one
                of the tuning methods (e.g. tune_hyperparams_bayes_bfgs).
                If supplied, must be a numpy array of shape (N, 2) where
                N is the number of hyperparams for the kernel in question.
            method (str): one of "srht", "srht_2" or "gauss". srht is MUCH faster for
                large datasets and should always be preferred to "gauss".
                "srht_2" runs two passes over the dataset. For the same max_rank,
                the preconditioner built by "srht_2" will generally reduce the
                number of CG iterations by 25-30% compared with a preconditioner
                built by "gauss" or "srht", but it does of course incur the
                additional expense of a second pass over the dataset.

        Returns:
            preconditioner: A preconditioner object.
            achieved_ratio (float): The min eigval of the preconditioner over
                lambda, the noise hyperparameter shared between all kernels.
                This value has decent predictive value for assessing how
                well the preconditioner is likely to perform.
        """
        self._run_pre_fitting_prep(dataset, preset_hyperparams, max_rank)
        if self.device == "gpu":
            preconditioner = Cuda_RandNysPreconditioner(self.kernel, dataset, max_rank,
                        self.verbose, self.random_seed, method)
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
        else:
            preconditioner = CPU_RandNysPreconditioner(self.kernel, dataset, max_rank,
                        self.verbose, self.random_seed, method)
        self._run_post_fitting_cleanup(dataset)
        return preconditioner, preconditioner.achieved_ratio



    def exact_nmll(self, hyperparams, dataset):
        """Calculates the exact negative marginal log likelihood (the model
        'score') using matrix decompositions. Fast for small numbers of random
        features but poor scaling to larger numbers. Can be numerically unstable
        for extreme hyperparameter values, since it uses a cholesky decomposition.

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
        self._run_post_nmll_cleanup(dataset)
        return negloglik



    def exact_nmll_gradient(self, hyperparams, dataset, subsample = 1):
        """Calculates the gradient of the negative marginal log likelihood w/r/t
        the hyperparameters for a specified set of hyperparameters using
        exact methods (matrix decompositions). Fast for small numbers of
        random features, very slow for large. Can be numerically unstable
        for extreme hyperparameter values, since it uses a cholesky decomposition.

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
        except Exception as err:
            return constants.DEFAULT_SCORE_IF_PROBLEM, hyperparams - init_hparams

        #Same problem as above, only occasionally the LAPACK routines return
        #nan instead of raising an error.
        if np.isnan(negloglik):
            return constants.DEFAULT_SCORE_IF_PROBLEM, hyperparams - init_hparams
        self._run_post_nmll_cleanup(dataset)
        return float(negloglik), grad



    def approximate_nmll(self, hyperparams, dataset, max_rank=1024,
            nsamples=25, niter=500, tol=1e-6,
            preconditioner_mode="srht_2"):
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
            max_rank (int): The preconditioner rank for approximate NMLL estimation.
                A larger value may reduce the number of iterations for nmll approximation
                to converge and improve estimation accuracy, but will also increase
                cost for preconditioner construction. This needs to be increased
                (e.g. to 2048) if the data is close to noise-free.
            nsamples (int): The number of probes for approximate NMLL estimation.
                A larger value may improve accuracy of estimation but with increased
                computational cost. This rarely needs to be adjusted -- the default
                is usually fine.
            niter (int): The maximum number of iterations for approximate NMLL.
                This rarely needs to be adjusted -- the default is usually fine.
            tol (float): The convergence tolerance for approximate NMLL.
                A smaller value may improve accuracy of estimation but with
                increased computational cost. This rarely needs to be adjusted --
                the default is usually fine.
            preconditioner_mode (str): One of "srht", "srht_2". Determines the mode of
                preconditioner construction. "srht_2" is recommended.

        Returns:
            nmll (float): The negative marginal log likelihood for the
                input hyperparameters.
            inv_trace (float): The approximate inverse of the trace. Only
                returned if retrieve_trace is True.
        """
        self._run_singlepoint_nmll_prep(dataset, exact_method = False,
            nmll_rank = max_rank)
        if self.kernel is None:
            raise ValueError("Must call self.initialize before calculating NMLL.")
        self.weights, self.var = None, None

        self.kernel.set_hyperparams(hyperparams, logspace = True)
        preconditioner = None
        if self.verbose:
            print("Now building preconditioner...")

        if max_rank > 0:
            preconditioner = RandNysTuningPreconditioner(self.kernel, dataset, max_rank,
                        False, self.random_seed, preconditioner_mode)
            if preconditioner.achieved_ratio > 250 and 2 * max_rank < self.kernel.get_num_rffs():
                rank = 2 * max_rank
                preconditioner = RandNysTuningPreconditioner(self.kernel, dataset, rank,
                        False, self.random_seed, preconditioner_mode)

        if self.verbose:
            print("Now fitting...")

        if self.device == "gpu":
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
            cg_operator = GPU_ConjugateGrad()
            resid = cp.zeros((self.kernel.get_num_rffs(), 2, nsamples + 1),
                            dtype = cp.float64)
            probes = generate_normal_probes_gpu(nsamples, self.kernel.get_num_rffs(),
                        self.random_seed, preconditioner)
        else:
            cg_operator = CPU_ConjugateGrad()
            resid = np.zeros((self.kernel.get_num_rffs(), 2, nsamples + 1),
                            dtype = np.float64)
            probes = generate_normal_probes_cpu(nsamples, self.kernel.get_num_rffs(),
                        self.random_seed, preconditioner)

        if preconditioner is None:
            z_trans_y, y_trans_y = calc_zty(dataset, self.kernel)
        else:
            z_trans_y = preconditioner.get_zty()
            y_trans_y = preconditioner.get_yty()

        resid[:,0,0] = z_trans_y / dataset.get_ndatapoints()
        resid[:,0,1:] = probes

        x_k, alphas, betas = cg_operator.fit(dataset, self.kernel,
                    preconditioner, resid, niter, tol, verbose = False,
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

        self._run_post_nmll_cleanup(dataset)
        return negloglik


    def fit(self, dataset, preconditioner = None,
                tol = 1e-6, preset_hyperparams=None, max_iter = 500,
                run_diagnostics = False, mode = "cg",
                suppress_var = False):
        """Fits the model after checking that the input data
        is consistent with the kernel choice and other user selections.

        Args:
            dataset: Object of class OnlineDataset or OfflineDataset.
            preconditioner: Either None or a valid Preconditioner (e.g.
                CudaRandomizedPreconditioner, CPURandomizedPreconditioner
                etc). If None, no preconditioning is used.
            tol (float): The threshold below which iterative strategies (L-BFGS, CG,
                SGD) are deemed to have converged. Defaults to 1e-5. Note that how
                reaching the threshold is assessed may depend on the algorithm.
            preset_hyperparams: Either None or a numpy array. If None,
                hyperparameters must already have been tuned using one
                of the tuning methods (e.g. tune_hyperparams_bayes_bfgs).
                If supplied, must be a numpy array of shape (N, 2) where
                N is the number of hyperparams for the kernel in question.
            max_iter (int): The maximum number of epochs for iterative strategies.
            run_diagnostics (bool): If True, the number of conjugate
                gradients and the preconditioner diagnostics ratio are returned.
            mode (str): Must be one of "cg", "lbfgs", "exact".
                Determines the approach used. If 'exact', self.kernel.get_num_rffs
                must be <= constants.constants.MAX_CLOSED_FORM_RFFS.
            suppress_var (bool): If True, do not calculate variance. This is generally only
                useful when optimizing hyperparameters, since otherwise we want to calculate
                the variance. It is best to leave this as default False unless performing
                hyperparameter optimization.

        Returns:
            Does not return anything unless run_diagnostics is True.
            n_iter (int): The number of iterations if applicable.
            losses (list): The loss on each iteration. Only for SGD and CG, otherwise,
                empty list.

        Raises:
            ValueError: The input dataset is checked for validity before tuning is
                initiated, an error is raised if problems are found."""
        self._run_pre_fitting_prep(dataset, preset_hyperparams)
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
            if run_diagnostics:
                self.weights, n_iter, losses = cg_fit_lib_internal(self.kernel, dataset, tol,
                    max_iter, preconditioner, self.verbose)
            else:
                self.weights, n_iter, losses = cg_fit_lib_ext(self.kernel, dataset, tol,
                    max_iter, preconditioner, self.verbose)

        elif mode == "lbfgs":
            model_fitter = lBFGSModelFit(dataset, self.kernel,
                    self.device, self.verbose)
            self.weights, n_iter, losses = model_fitter.fit_model_lbfgs(max_iter, tol,
                    preconditioner)

        else:
            raise ValueError("Unrecognized fitting mode supplied. Must provide one of "
                        "'lbfgs', 'cg', 'exact'.")
        if not suppress_var:
            if self.verbose:
                print("Now performing variance calculations...")
            #TODO: This is a little bit of a hack; we use exact variance calc
            #UNLESS we are dealing with a linear kernel with a very large number
            #of input features. Find a better / more satisfactory way to resolve this.
            if self.kernel_choice in ["Linear", "ExactQuadratic"]:
                if self.kernel.get_num_rffs() > constants.MAX_VARIANCE_RFFS:
                    self.var = InterDevicePreconditioner(self.kernel, dataset,
                        self.variance_rffs, False, self.random_seed, "srht")
                    self.exact_var_calculation = False
                else:
                    self.variance_rffs = self.kernel.get_num_rffs()
                    self.var = calc_variance_exact(self.kernel, dataset, self.kernel_choice,
                                self.variance_rffs)
            else:
                self.var = calc_variance_exact(self.kernel, dataset, self.kernel_choice,
                                self.variance_rffs)


        if self.verbose:
            print("Fitting complete.")
        if self.device == "gpu":
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
        self._run_post_fitting_cleanup(dataset)

        if run_diagnostics:
            return n_iter, losses



    #############################
    #The next block of functions are used for tuning hyperparameters. We provide
    #a couple of strategies and of course the user can easily create their own
    #(e.g. using Optuna).
    #############################


    def tune_hyperparams_crude(self, dataset, bounds = None, random_seed = 123,
                    max_bayes_iter = 30, subsample = 1):
        """Tunes the hyperparameters using Bayesian optimization, but with
        a 'trick' that simplifies the problem greatly for 2-4 hyperparameter
        kernels. Hyperparameters are scored using an exact NMLL calculation.
        The NMLL calculation uses a matrix decomposition with cubic
        scaling in the number of random features, so it is extremely slow
        for anything more than 3-4,000 random features, but has
        low risk of overfitting and is easy to use. It is therefore
        intended as a "quick-and-dirty" method. We recommend using
        this with a small number of random features (e.g. 500 - 3000)
        and if performance is insufficient, fine-tune hyperparameters
        using a method with better scalability.

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
            scores (tuple): A tuple where the first element is the sigma values
                evaluated and the second is the resulting scores. Can be useful
                for diagnostic purposes.

        Raises:
            ValueError: The input dataset is checked for validity before tuning is
                initiated. If problems are found, a raise exception will provide an
                explanation of the error. This method will also raise an exception
                if you try to use it on a kernel with > 4 hyperparameters (since
                this strategy no longer provides any benefit under those conditions)
                or < 3.
            n_init_pts (int): The number of initial grid points to evaluate before
                Bayesian optimization. 10 (the default) is usually fine. If you are
                searcing a smaller space, however, you can save time by using
                a smaller # (e.g. 5).
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

        self._run_post_nmll_cleanup(dataset, hyperparams)
        return hyperparams, n_feval, best_score


    def tune_hyperparams_direct(self, dataset, bounds = None,
                max_iter = 50, tol = 1, tuning_method = "Powell",
                starting_hyperparams = None, n_restarts = 1,
                nmll_method = "exact", nmll_rank=1024, nmll_probs=25,
                nmll_iter=500, nmll_tol=1e-6, preconditioner_mode="srht_2"):
        """Tunes the hyperparameters WITHOUT using gradient information.
        The methods included here can be extremely efficient for one-
        and two-hyperparameter kernels but are not recommended for kernels
        with many hyperparameters. This method can either use exact
        NMLL (small num_rffs) or approximate (scales to larger num_rffs, but
        approximation is only good so long as settings selected are reasonable).

        Args:
            dataset: Object of class OnlineDataset or OfflineDataset.
            bounds (np.ndarray): The bounds for optimization. Must be supplied,
                in contrast to most other tuning routines, since this routine
                is more seldom used for searching the whole hyperparameter space.
                Must be a 2d numpy array of shape (num_hyperparams, 2).
            max_iter (int): The maximum number of iterations.
            tol (float): Criteria for convergence.
            tuning_method (str): One of 'Powell', 'Nelder-Mead'.
                Nelder-Mead is usually better than Powell but may
                take many more iterations to converge.
            starting_hyperparams: Either None or a numpy array of shape (nhparams)
                where (nhparams) is the number of hyperparameters for the selected
                kernel. If None, a starting point is randomly selected using
                random_seed.
            n_restarts (int): The number of times to restart the optimizer.
                Ignored if starting hyperparameters are supplied.
            nmll_method (str): One of 'exact', 'approximate'. 'Exact' is fast
                and accurate if num_rffs is small but scales poorly to large
                num_rffs. 'approximate' has better scaling but is only accurate
                so long as the `approx_nmll_settings` are reasonable.
            nmll_rank (int): The preconditioner rank for approximate NMLL estimation.
                A larger value may reduce the number of iterations for nmll approximation
                to converge and improve estimation accuracy, but will also increase
                cost for preconditioner construction. This needs to be increased
                (e.g. to 2048) if the data is close to noise-free.
            nmll_probes (int): The number of probes for approximate NMLL estimation.
                A larger value may improve accuracy of estimation but with increased
                computational cost. This rarely needs to be adjusted -- the default
                is usually fine.
            nmll_iter (int): The maximum number of iterations for approximate NMLL.
                This rarely needs to be adjusted -- the default is usually fine.
            nmll_tol (float): The convergence tolerance for approximate NMLL.
                A smaller value may improve accuracy of estimation but with
                increased computational cost. This rarely needs to be adjusted --
                the default is usually fine.
            preconditioner_mode (str): One of "srht", "srht_2". Determines the mode of
                preconditioner construction. "srht_2" is recommended.

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
        else:
            raise ValueError("Invalid tuning method supplied.")

        if nmll_method == "approximate":
            optim_bounds = self._run_pre_nmll_prep(dataset, bounds, nmll_rank)
            args = (dataset, nmll_rank, nmll_probs, nmll_iter,
                    nmll_tol, preconditioner_mode)
            cost_fun = self.approximate_nmll
        elif nmll_method == "exact":
            optim_bounds = self._run_pre_nmll_prep(dataset, bounds)
            args = (dataset,)
            cost_fun = self.exact_nmll
        else:
            raise ValueError("Invalid nmll method supplied.")


        bounds_tuples = list(map(tuple, optim_bounds))
        rng = np.random.default_rng(self.random_seed)

        if starting_hyperparams is None:
            x0 = self.kernel.get_hyperparams()
            n_repeats = n_restarts
        elif isinstance(starting_hyperparams, np.ndarray) and \
            starting_hyperparams.shape[0] == self.kernel.get_hyperparams.shape[0]:
            x0 = starting_hyperparams
            n_repeats = 1
        else:
            raise ValueError("Invalid starting hyperparams were supplied.")

        best_score, n_feval = np.inf, 0

        for _ in range(n_repeats):
            res = minimize(cost_fun, x0 = x0,
                    options=options, method=tuning_method,
                    args = args, bounds = bounds_tuples)
            n_feval += res.nfev

            if res.fun < best_score:
                n_feval, hyperparams, best_score = res.nfev, res.x, res.fun
            x0 = [rng.uniform(low = optim_bounds[j,0],
                    high = optim_bounds[j,1]) for j in range(optim_bounds.shape[0])]
            x0 = np.asarray(x0)

        return hyperparams, n_feval, best_score



    def tune_hyperparams_lbfgs(self, dataset,
            max_iter = 50, n_restarts = 1, starting_hyperparams = None,
            bounds = None):
        """Tunes the hyperparameters using the L-BFGS algorithm, with
        NMLL as the objective.
        It uses either a supplied set of starting hyperparameters OR
        randomly chosen locations. If the latter, it is run
        n_restarts times. Because it uses exact NMLL rather than
        approximate, this method is only suitable to small numbers
        of random features; scaling to larger numbers of random
        features (e.g. > 5000) is poor. Nonetheless, this can be
        very effective, especially for kernels with > 2 hyperparameters,
        where the direct methods tend to be less efficient.

        Args:
            dataset: Object of class OnlineDataset or OfflineDataset.
            max_iter (int): The maximum number of iterations for
                which l-bfgs should be run per restart.
            n_restarts (int): The maximum number of restarts to run l-bfgs.
            starting_hyperparams (np.ndarray): A starting point for l-bfgs
                based optimization. Defaults to None. If None, randomly
                selected locations are used.
            bounds (np.ndarray): The bounds for optimization. Defaults to
                None, in which case the kernel uses its default bounds.
                If supplied, must be a 2d numpy array of shape (num_hyperparams, 2).

        Returns:
            hyperparams (np.ndarray): The best hyperparams found during optimization.
            n_feval (int): The number of function evaluations during optimization.
            best_score (float): The best negative marginal log-likelihood achieved.

        Raises:
            ValueError: The input dataset is checked for validity before tuning is
                initiated. If problems are found, a ValueError will provide an
                explanation of the error.
        """
        optim_bounds = self._run_pre_nmll_prep(dataset, bounds)

        init_hyperparams = starting_hyperparams
        if init_hyperparams is None:
            init_hyperparams = self.kernel.get_hyperparams(logspace=True)

        best_x, best_score, net_iterations = None, np.inf, 0
        bounds_tuples = list(map(tuple, optim_bounds))

        rng = np.random.default_rng(self.random_seed)
        args, cost_fun = (dataset,), self.exact_nmll_gradient

        for iteration in range(n_restarts):
            res = minimize(cost_fun, options={"maxiter":max_iter},
                        x0 = init_hyperparams, args = args,
                        jac = True, bounds = bounds_tuples)

            net_iterations += res.nfev
            if res.fun < best_score:
                best_x = res.x
                best_score = res.fun
            if self.verbose:
                print(f"Restart {iteration} completed. Best score is {best_score}.")
            init_hyperparams = [rng.uniform(low = optim_bounds[j,0], high = optim_bounds[j,1])
                    for j in range(optim_bounds.shape[0])]
            init_hyperparams = np.asarray(init_hyperparams)


        if best_x is None:
            raise ValueError("All restarts failed to find acceptable hyperparameters.")

        self._run_post_nmll_cleanup(dataset, best_x)
        return best_x, net_iterations, best_score
