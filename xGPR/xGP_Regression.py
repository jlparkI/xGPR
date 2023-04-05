"""Describes the xGPRegression class.

The xGPRegression class provides the tools needed to fit a regression
model and make predictions for new datapoints. It inherits from
GPRegressionBaseclass.
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
from .regression_baseclass import GPRegressionBaseclass
from .preconditioners.rand_nys_preconditioners import CPU_RandNysPreconditioner
from .preconditioners.tuning_preconditioners import RandNysTuningPreconditioner
from .preconditioners.inter_device_preconditioners import InterDevicePreconditioner

from .optimizers.pure_bayes_optimizer import pure_bayes_tuning
from .optimizers.bayes_grid_optimizer import bayes_grid_tuning
from .optimizers.lb_optimizer import shared_hparam_search
from .optimizers.crude_grid_optimizer import crude_grid_tuning

from .fitting_toolkit.lbfgs_fitting_toolkit import lBFGSModelFit
from .fitting_toolkit.sgd_fitting_toolkit import sgdModelFit
from .fitting_toolkit.ams_grad_toolkit import amsModelFit
from .fitting_toolkit.cg_fitting_toolkit import cg_fit_lib_ext
from .fitting_toolkit.exact_fitting_toolkit import calc_weights_exact, calc_variance_exact

from .scoring_toolkit.approximate_nmll_calcs import estimate_logdet, estimate_nmll
from .scoring_toolkit.nmll_gradient_tools import exact_nmll_reg_grad, calc_gradient_terms
from .scoring_toolkit.probe_generators import generate_normal_probes_gpu
from .scoring_toolkit.probe_generators import generate_normal_probes_cpu
from .scoring_toolkit.exact_nmll_calcs import calc_zty, calc_design_mat, direct_weight_calc

from cg_tools import CPU_ConjugateGrad, GPU_ConjugateGrad



class xGPRegression(GPRegressionBaseclass):
    """A subclass of GPRegressionBaseclass that houses methods
    unique to regression problems. It does not have
    any attributes unique to it aside from those
    of the parent class."""

    def __init__(self, training_rffs, fitting_rffs, variance_rffs = 16,
                    kernel_choice="rbf",
                    device = "cpu",
                    kernel_specific_params = constants.DEFAULT_KERNEL_SPEC_PARMS,
                    verbose = True,
                    num_threads = 2,
                    double_precision_fht = False):
        """The constructor for xGPRegression. Passes arguments onto
        the parent class constructor.

        Args:
            training_rffs (int): The number of random Fourier features
                to use for hyperparameter tuning.
            fitting_rffs (int): The number of random Fourier features
                to use for posterior predictive mean (i.e. the predicted
                value for new datapoints).
            variance_rffs (int): The number of random Fourier features
                to use for posterior predictive variance (i.e. calculating
                uncertainty on predictions). Defaults to 64.
            kernel_choice (str): The kernel that the model will use.
                Must be one of constants.ACCEPTABLE_KERNELS.
                Defaults to 'rbf'.
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
            double_precision_fht (bool): If True, use double precision during FHT for
                generating random features. For most problems, it is not beneficial
                to set this to True -- it merely increases computational expense
                with negligible benefit -- but this option is useful for testing.
                Defaults to False.
        """
        super().__init__(training_rffs, fitting_rffs, variance_rffs,
                        kernel_choice, device = device,
                        kernel_specific_params = kernel_specific_params,
                        verbose = verbose, num_threads = num_threads,
                        double_precision_fht = double_precision_fht)




    def predict(self, input_x, get_var = True,
            chunk_size = 2000):
        """Generate a predicted value for each
        input datapoint -- and if desired the variance.

        Args:
            input_x (np.ndarray): The input data. Should be a 2d numpy
                array (if non-convolution kernel) or 3d (if convolution
                kernel).
            get_var (bool): If True, return (predictions, variance).
                If False, return predictions. Defaults to True.
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
                pred_var = lambda_**2 * ((xfeatures * pred_var).sum(axis=1))
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
        return preds * self.trainy_std + self.trainy_mean, var * self.trainy_std**2




    def build_preconditioner(self, dataset, max_rank = 512,
                        preset_hyperparams = None, random_state = 123,
                        method = "srht"):
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
            random_state (int): Seed for the random number generator.
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
        self._run_fitting_prep(dataset, random_state, preset_hyperparams)
        if max_rank < 1:
            raise ValueError("Invalid value for max_rank.")
        if max_rank >= self.fitting_rffs:
            raise ValueError("Max rank should be < self.fitting_rffs.")

        if self.device == "gpu":
            preconditioner = Cuda_RandNysPreconditioner(self.kernel, dataset, max_rank,
                        self.verbose, random_state, method)
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
        else:
            preconditioner = CPU_RandNysPreconditioner(self.kernel, dataset, max_rank,
                        self.verbose, random_state, method)
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
        self.kernel.set_hyperparams(hyperparams, logspace=True)
        nsamples = dataset.get_ndatapoints()
        z_trans_z, z_trans_y, y_trans_y = calc_design_mat(dataset, self.kernel)
        if self.device == "cpu":
            diagfunc, logfunc = np.diag, np.log
        else:
            diagfunc, logfunc = cp.diag, cp.log

        lambda_p = self.kernel.get_lambda()
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

        negloglik = (-0.5 / lambda_p**2) * (z_trans_y.T @ weights - y_trans_y)
        negloglik += (nsamples - z_trans_z.shape[0]) * logfunc(lambda_p)
        negloglik += nsamples * 0.5 * logfunc(2 * np.pi)
        negloglik += logfunc(diagfunc(chol_z_trans_z)).sum()
        negloglik = float(negloglik)

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
        init_hparams = self.kernel.get_hyperparams()
        self.kernel.set_hyperparams(hyperparams, logspace=True)
        hparams = self.kernel.get_hyperparams(logspace=False)

        if self.verbose:
            print("Evaluating gradient...")

        z_trans_z, z_trans_y, y_trans_y, dz_dsigma_ty, inner_deriv, nsamples = \
                        calc_gradient_terms(dataset, self.kernel, self.device, subsample)
        grad = np.zeros((hparams.shape[0]))

        #Try-except here since very occasionally, optimizer samples a really
        #terrible set of hyperparameters that with numerical error has resulted in a
        #non-positive-definite design matrix. TODO: Find a good workaround to
        #avoid this whole problem.
        try:
            chol_z_trans_z, weights, grad = exact_nmll_reg_grad(z_trans_z, z_trans_y,
                        y_trans_y, hparams, nsamples, dz_dsigma_ty, inner_deriv,
                        self.device)
        except Exception as err:
            return constants.DEFAULT_SCORE_IF_PROBLEM, hyperparams - init_hparams

        #Note that in the following, hparams[0] and hparams[1] are lambda_ and
        #beta_, the two hyperparameters shared between all kernels.
        negloglik = (-0.5 / hparams[0]**2) * (z_trans_y.T @ weights)

        if self.device == "cpu":
            negloglik += np.log(np.diag(chol_z_trans_z)).sum()
            negloglik += (nsamples - chol_z_trans_z.shape[0]) * np.log(hparams[0])
        else:
            negloglik += cp.log(cp.diag(chol_z_trans_z)).sum()
            negloglik += (nsamples - chol_z_trans_z.shape[0]) * cp.log(hparams[0])

        negloglik = float(0.5 * y_trans_y / hparams[0]**2 + negloglik)
        negloglik += nsamples * 0.5 * np.log(2 * np.pi)

        #Same problem as above, only occasionally the LAPACK routines return
        #nan instead of raising an error.
        if np.isnan(negloglik):
            return constants.DEFAULT_SCORE_IF_PROBLEM, hyperparams - init_hparams
        return float(negloglik), grad


    def approximate_nmll(self, hyperparams, dataset, max_rank = 1024,
            nsamples = 25, random_seed = 123, niter = 1000,
            tol = 1e-6, pretransform_dir = None,
            preconditioner_mode = "srht_2"):
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
            max_rank: An int >= 0. If 0, no preconditioning is used,
                which may greatly decrease the accuracy of the estimates for
                hyperparameters that result in an ill-conditioned Z^T Z.
                Otherwise, this number is used as max_rank to build a
                preconditioner.
            nsamples (int): The number of samples to draw for Hutchison trace
                estimation. More improves the accuracy of the estimate but
                increases computational expense.
            random_seed (int): A seed for random number generation.
            niter (int): The maximum number of iterations for SLQ. More increases
                the accuracy of the estimate but increases computational
                expense.
            tol (float): If tol is reached, iterations will be stopped early.
            pretransform_dir (str): Either None or a valid filepath where pretransformed
                data can be saved. If not None, the dataset is "pretransformed" before
                each round of fitting. This can take up a lot of disk space if the number
                of random features is large, but can greatly increase speed of fitting
                for convolution kernels with large # random features.
            preconditioner_mode (str): One of "srht", "srht_2". Determines the mode of
                preconditioner construction. "srht" is cheaper, requiring one pass over
                the dataset, but lower quality. "srht_2" requires two passes over the
                dataset but is better. Prefer "srht_2" unless you are running on CPU,
                in which case "srht" may be preferable.

        Returns:
            nmll (float): The negative marginal log likelihood for the
                input hyperparameters.
            inv_trace (float): The approximate inverse of the trace. Only
                returned if retrieve_trace is True.
        """
        self.kernel.set_hyperparams(hyperparams, logspace = True)
        if max_rank >= self.kernel.get_num_rffs():
            raise ValueError("For calculating approximate NMLL with a preconditioner, "
                    "the specified preconditioner rank must be < the number of random "
                    "features.")

        train_dataset = dataset
        if pretransform_dir is not None:
            train_dataset = self._pretransform_dataset(dataset, pretransform_dir)
            train_dataset.device = self.kernel.device

        preconditioner = None
        if self.verbose:
            print("Now building preconditioner...")

        if max_rank > 0:
            preconditioner = RandNysTuningPreconditioner(self.kernel, train_dataset, max_rank,
                        False, random_seed, preconditioner_mode)
            if preconditioner.achieved_ratio > 250 and 2 * max_rank < self.kernel.get_num_rffs():
                rank = 2 * max_rank
                preconditioner = RandNysTuningPreconditioner(self.kernel, train_dataset, rank,
                        False, random_seed, preconditioner_mode)

        if self.verbose:
            print("Now fitting...")

        if self.device == "gpu":
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
            cg_operator = GPU_ConjugateGrad()
            resid = cp.zeros((self.kernel.get_num_rffs(), 2, nsamples + 1),
                            dtype = cp.float64)
            probes = generate_normal_probes_gpu(nsamples, self.kernel.get_num_rffs(),
                        random_seed, preconditioner)
        else:
            cg_operator = CPU_ConjugateGrad()
            resid = np.zeros((self.kernel.get_num_rffs(), 2, nsamples + 1),
                            dtype = np.float64)
            probes = generate_normal_probes_cpu(nsamples, self.kernel.get_num_rffs(),
                        random_seed, preconditioner)

        if preconditioner is None:
            z_trans_y, y_trans_y = calc_zty(train_dataset, self.kernel)
        else:
            z_trans_y = preconditioner.get_zty()
            y_trans_y = preconditioner.get_yty()

        resid[:,0,0] = z_trans_y / dataset.get_ndatapoints()
        resid[:,0,1:] = probes

        x_k, alphas, betas = cg_operator.fit(train_dataset, self.kernel,
                    preconditioner, resid, niter, tol, verbose = False,
                    nmll_settings = True)

        x_k[:,0] *= dataset.get_ndatapoints()

        logdet = estimate_logdet(alphas, betas, self.kernel.get_num_rffs(),
                        preconditioner, self.device)
        nmll = estimate_nmll(train_dataset, self.kernel, logdet, x_k,
                        z_trans_y, y_trans_y)
        if self.verbose:
            print("NMLL evaluation completed.")
        if pretransform_dir is not None:
            train_dataset.delete_dataset_files()
        return float(nmll)



    def fit(self, dataset, preconditioner = None,
                tol = 1e-6, preset_hyperparams=None, max_iter = 500,
                random_seed = 123, run_diagnostics = False,
                mode = "cg", suppress_var = False,
                manual_lr = None):
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
            random_seed (int): The random seed for the random number generator.
            run_diagnostics (bool): If True, the number of conjugate
                gradients and the preconditioner diagnostics ratio are returned.
            mode (str): Must be one of "sgd", "amsgrad", "cg", "lbfgs", "exact".
                Determines the approach used. If 'exact', self.kernel.get_num_rffs
                must be <= constants.constants.MAX_CLOSED_FORM_RFFS.
            suppress_var (bool): If True, do not calculate variance. This is generally only
                useful when optimizing hyperparameters, since otherwise we want to calculate
                the variance. It is best to leave this as default False unless performing
                hyperparameter optimization.
            manual_lr (float): Either None or a float. If not None, this is the initial
                learning rate used for stochastic gradient descent or ams grad (ignored
                for all other fitting modes). If None, the algorithm will try to determine
                a good initial learning rate itself.

        Returns:
            Does not return anything unless run_diagnostics is True.
            n_iter (int): The number of iterations for conjugate gradients, L-BFGS or sgd.
            losses (list): The loss on each iteration. Only for SGD and CG, otherwise,
                empty list.

        Raises:
            ValueError: The input dataset is checked for validity before tuning is
                initiated, an error is raised if problems are found."""
        self._run_fitting_prep(dataset, random_seed, preset_hyperparams)
        if self.verbose:
            print("starting fitting")

        if mode == "exact":
            if self.kernel.get_num_rffs() > constants.MAX_CLOSED_FORM_RFFS:
                raise ValueError("You specified 'exact' fitting, but self.fitting_rffs is "
                        f"> {constants.MAX_CLOSED_FORM_RFFS}.")
            self.weights, n_iter, losses = calc_weights_exact(dataset, self.kernel)

        elif mode == "cg":
            self.weights, n_iter, losses = cg_fit_lib_ext(self.kernel, dataset, tol,
                    max_iter, preconditioner, self.verbose)

        elif mode == "lbfgs":
            model_fitter = lBFGSModelFit(dataset, self.kernel,
                    self.device, self.verbose)
            self.weights, n_iter, losses = model_fitter.fit_model_lbfgs(max_iter, tol)

        elif mode == "sgd":
            model_fitter = sgdModelFit(self.kernel.get_lambda(), self.device, self.verbose)
            self.weights, n_iter, losses = model_fitter.fit_model(dataset,
                    self.kernel, tol = tol, max_epochs = max_iter,
                    preconditioner = preconditioner, manual_lr = manual_lr)

        elif mode == "amsgrad":
            model_fitter = amsModelFit(self.kernel.get_lambda(), self.device, self.verbose)
            self.weights, n_iter, losses = model_fitter.fit_model(dataset,
                    self.kernel, tol = tol, max_epochs = max_iter)

        else:
            raise ValueError("Unrecognized fitting mode supplied. Must provide one of "
                        "'lbfgs', 'cg', 'sgd', 'amsgrad', 'exact'.")
        if not suppress_var:
            if self.verbose:
                print("Now performing variance calculations...")
            #TODO: This is a little bit of a hack; we use exact variance calc
            #UNLESS we are dealing with a linear kernel with a very large number
            #of input features. Find a better / more satisfactory way to resolve this.
            if self.kernel_choice == "Linear":
                if self.kernel.get_num_rffs() > constants.MAX_VARIANCE_RFFS:
                    self.var = InterDevicePreconditioner(self.kernel, dataset,
                        self.variance_rffs, False, random_seed, "srht")
                    self.exact_var_calculation = False
                else:
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
        if run_diagnostics:
            return n_iter, losses



    #############################
    #The next block of functions are used for tuning hyperparameters. We provide
    #a variety of strategies (the user can additionally combine them
    #to generate more), with some strong recommendations on which to use when
    #(see docs). The remaining code in this class is dedicated to these
    #various strategies.
    #############################



    def tune_hyperparams_fine_bayes(self, dataset, bounds = None, random_seed = 123,
                    max_bayes_iter = 30, tol = 1e-1, nmll_rank = 1024,
                    nmll_probes = 25, nmll_iter = 500,
                    nmll_tol = 1e-6, pretransform_dir = None,
                    preconditioner_mode = "srht_2"):
        """Tunes the hyperparameters using Bayesian optimization, WITHOUT
        using gradient information. This algorithm is not very efficient for searching the entire
        hyperparameter space, BUT for 3 and 4 hyperparameter kernels, it can be very effective
        for searching some bounded region. Consequently, it may sometimes be useful to
        use a "crude" method (e.g. minimal_bayes) to find a starting point,
        then run this method to fine-tune.

        This method uses approximate NMLL. Note that the approximation will only be
        good so long as the 'nmll' settings selected are reasonable.

        Args:
            dataset: Object of class OnlineDataset or OfflineDataset.
            bounds (np.ndarray): The bounds for optimization. Must be supplied,
                in contrast to most other tuning routines, since this routine
                is more seldom used for searching the whole hyperparameter space.
                Must be a 2d numpy array of shape (num_hyperparams, 2).
            random_seed (int): A random seed for the random
                number generator. Defaults to 123.
            max_bayes_iter (int): The maximum number of iterations of Bayesian
                optimization.
            tol (float): Criteria for convergence.
            nmll_rank (int): The preconditioner rank for approximate NMLL estimation.
                A larger value may reduce the number of iterations for nmll approximation
                to converge and improve estimation accuracy, but will also increase
                cost for preconditioner construction.
            nmll_probes (int): The number of probes for approximate NMLL estimation.
                A larger value may improve accuracy of estimation but with increased
                computational cost.
            nmll_iter (int): The maximum number of iterations for approximate NMLL.
                A larger value may improve accuracy of estimation but with
                increased computational cost.
            nmll_tol (float): The convergence tolerance for approximate NMLL.
                A smaller value may improve accuracy of estimation but with
                increased computational cost.
            pretransform_dir (str): Either None or a valid filepath where pretransformed
                data can be saved. If not None, the dataset is "pretransformed" before
                each round of evaluations when using approximate NMLL. This can take up a
                lot of disk space if the number of random features is large, but can
                greatly increase speed of fitting for convolution kernels with large #
                random features.
            preconditioner_mode (str): One of "srht", "srht_2". Determines the mode of
                preconditioner construction. "srht" is cheaper, requiring one pass over
                the dataset, but lower quality. "srht_2" requires two passes over the
                dataset but is better. Prefer "srht_2" unless you are running on CPU,
                in which case "srht" may be preferable.

        Returns:
            hyperparams (np.ndarray): The best hyperparams found during optimization.
            n_feval (int): The number of function evaluations during optimization.
            best_score (float): The best negative marginal log-likelihood achieved.

        Raises:
            ValueError: The input dataset is checked for validity before tuning is
                initiated. If problems are found, a ValueError will provide an
                explanation of the error. This method will also raise a ValueError
                if you try to use it on a kernel with > 4 hyperparameters (since
                Bayesian tuning loses efficiency in high dimensions rapidly).
        """
        if nmll_rank >= self.training_rffs:
            raise ValueError("NMLL rank must be < the number of training rffs.")
        bounds = self._run_pretuning_prep(dataset, random_seed, bounds, "approximate")
        nmll_params = (nmll_rank, nmll_probes, random_seed,
                    nmll_iter, nmll_tol, pretransform_dir,
                    preconditioner_mode)

        hyperparams, best_score, n_feval = pure_bayes_tuning(self.approximate_nmll,
                        dataset, bounds, random_seed,
                        max_iter = max_bayes_iter,
                        verbose = self.verbose, tol = tol,
                        nmll_params = nmll_params)
        self._post_tuning_cleanup(dataset, hyperparams)
        return hyperparams, n_feval, best_score




    def tune_hyperparams_crude_grid(self, dataset, bounds = None, random_seed = 123,
                    n_gridpoints = 30, n_pts_per_dim = 10, subsample = 1,
                    eigval_quotient = 1e6, min_eigval = 1e-5):
        """Tunes the hyperparameters using gridsearch, but with
        a 'trick' that simplifies the problem greatly for 2-3 hyperparameter
        kernels. Hyperparameters are scored using an exact NMLL calculation.
        The NMLL calculation uses a matrix decomposition with cubic
        scaling in the number of random features, so it is extremely slow
        for anything more than 3-4,000 random features, but has
        low risk of overfitting and is easy to use. It is therefore
        intended as a "quick-and-dirty" method. We recommend using
        this with a small number of random features (e.g. 500 - 3000)
        as a starting point for fine-tuning.

        Args:
            dataset: Object of class OnlineDataset or OfflineDataset.
            bounds (np.ndarray): The bounds for optimization. If None, default
                boundaries for the kernel will be used. Otherwise, must be an
                array of shape (# hyperparams, 2).
            random_seed (int): A random seed for the random
                number generator. Defaults to 123.
            n_gridpoints (int): The number of gridpoints per non-shared hparam.
            n_pts_per_dim (int): The number of grid points per shared hparam.
            subsample (float): A value in the range [0.01,1] that indicates what
                fraction of the training set to use each time the score is
                calculated (the same subset is used every time). In general, 1
                will give better results, but using a subsampled subset can be
                a fast way to find the (approximate) location of a good
                hyperparameter set.
            eigval_quotient (float): A value by which the largest singular value
                of Z^T Z is divided to determine the smallest acceptable eigenvalue
                of Z^T Z (singular vectors with eigenvalues smaller than this
                are discarded). Setting this to larger values will make this
                slightly more accurate, at the risk of numerical stability issues.
                In general, do not change this without good reason.
            min_eigval (float): If the largest singular value of Z^T Z divided by
                eigval_quotient is < min_eigval, min_eigval is used as the cutoff
                threshold instead. Setting this to smaller values will make this
                slightly more accurate, at the risk of numerical stability
                issues. In general, do not change this without good reason.

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
        bounds = self._run_pretuning_prep(dataset, random_seed, bounds, "exact")
        num_hparams = self.kernel.get_hyperparams().shape[0]
        if num_hparams == 2:
            best_score, hyperparams = shared_hparam_search(np.array([]), self.kernel,
                    dataset, bounds, n_pts_per_dim, 3, subsample = subsample)
            n_feval, scores = 1, ()
        elif num_hparams == 3:
            hyperparams, scores, best_score = crude_grid_tuning(self.kernel,
                                dataset, bounds, self.verbose,
                                n_gridpoints, subsample = subsample,
                                eigval_quotient = eigval_quotient,
                                min_eigval = min_eigval)
            n_feval = n_gridpoints

        else:
            raise ValueError("The crude grid procedure is only appropriate for "
                    "kernels with 2-3 hyperparameters.")

        self._post_tuning_cleanup(dataset, hyperparams)
        return hyperparams, n_feval, best_score, scores




    def tune_hyperparams_crude_bayes(self, dataset, bounds = None, random_seed = 123,
                    max_bayes_iter = 30, bayes_tol = 1e-1, n_pts_per_dim = 10,
                    n_cycles = 3, n_init_pts = 10, subsample = 1,
                    eigval_quotient = 1e6, min_eigval = 1e-5):
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
            bayes_tol (float): Criteria for convergence for Bayesian
                optimization.
            n_pts_per_dim (int): The number of grid points per shared hparam.
            n_cycles (int): The number of cycles of "telescoping" grid search
                to run. Increasing n_pts_per_dim and n_cycles usually only
                results in very small improvements in performance.
            subsample (float): A value in the range [0.01,1] that indicates what
                fraction of the training set to use each time the score is
                calculated (the same subset is used every time). In general, 1
                will give better results, but using a subsampled subset can be
                a fast way to find the (approximate) location of a good
                hyperparameter set.
            eigval_quotient (float): A value by which the largest singular value
                of Z^T Z is divided to determine the smallest acceptable eigenvalue
                of Z^T Z (singular vectors with eigenvalues smaller than this
                are discarded). Setting this to larger values will make this
                slightly more accurate, at the risk of numerical stability issues.
                In general, do not change this without good reason.
            min_eigval (float): If the largest singular value of Z^T Z divided by
                eigval_quotient is < min_eigval, min_eigval is used as the cutoff
                threshold instead. Setting this to smaller values will make this
                slightly more accurate, at the risk of numerical stability
                issues. In general, do not change this without good reason.

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
        bounds = self._run_pretuning_prep(dataset, random_seed, bounds, "exact")
        num_hparams = self.kernel.get_hyperparams().shape[0]
        if num_hparams == 2:
            best_score, hyperparams = shared_hparam_search(np.array([]), self.kernel,
                    dataset, bounds, n_pts_per_dim, n_cycles, subsample = subsample)
            n_feval, scores = 1, ()
        elif 5 > num_hparams > 2:
            hyperparams, scores, best_score, n_feval = bayes_grid_tuning(self.kernel,
                                dataset, bounds, random_seed, max_bayes_iter,
                                self.verbose, bayes_tol,
                                n_pts_per_dim, n_cycles, n_init_pts,
                                subsample = subsample,
                                eigval_quotient = eigval_quotient,
                                min_eigval = min_eigval)

        else:
            raise ValueError("The crude_bayes procedure is only appropriate for "
                    "kernels with 3-4 hyperparameters.")

        self._post_tuning_cleanup(dataset, hyperparams)
        return hyperparams, n_feval, best_score, scores




    def tune_hyperparams_fine_direct(self, dataset, bounds = None,
                    optim_method = "Powell", starting_hyperparams = None,
                    random_seed = 123, max_iter = 50, nmll_rank = 1024,
                    nmll_probes = 25, nmll_iter = 500,
                    nmll_tol = 1e-6, pretransform_dir = None,
                    preconditioner_mode = "srht_2"):
        """Tunes hyperparameters using either Nelder-Mead or Powell (Powell
        preferred), with an approximate NMLL calculation instead of exact.
        This is generally not useful for searching the whole hyperparameter space.
        If given a good starting point, however, it can work pretty well.
        Consequently it is best used to fine-tune "crude" hyperparameters obtained
        from an initial tuning run with a small number of random features in a
        less scalable method (e.g. minimal bayes).

        This method uses approximate NMLL. Note that the approximation will only be
        good so long as the 'nmll' parameters selected are reasonable.

        Args:
            dataset: Object of class OnlineDataset or OfflineDataset.
            bounds (np.ndarray): The bounds for optimization.
                Either a 2d numpy array of shape (num_hyperparams, 2) or None,
                in which case default kernel boundaries are used.
            optim_method (str): One of "Powell", "Nelder-Mead".
            starting_hyperparams (np.ndarray): A starting point for optimization.
                Defaults to None. If None, randomly selected locations are used.
            random_seed (int): Seed for the random number generator.
            max_iter (int): The maximum number of iterations.
            nmll_rank (int): The preconditioner rank for approximate NMLL estimation.
                A larger value may reduce the number of iterations for nmll
                approximation to converge and improve estimation accuracy, but
                will also increase computational cost for preconditioner construction.
            nmll_probes (int): The number of probes for approximate NMLL estimation.
                A larger value may improve accuracy of estimation but with increased
                computational cost.
            nmll_iter (int): The maximum number of iterations for approximate NMLL.
                A larger value may improve accuracy of estimation but with increased
                computational cost.
            nmll_tol (float): The convergence tolerance for approximate NMLL.
                A smaller value may improve accuracy of estimation but with
                increased computational cost.
            pretransform_dir (str): Either None or a valid filepath where pretransformed
                data can be saved. If not None, the dataset is "pretransformed" before
                each round of evaluations when using approximate NMLL. This can take up a
                lot of disk space if the number of random features is large, but can
                greatly increase speed of fitting for convolution kernels with large #
                random features.
            preconditioner_mode (str): One of "srht", "srht_2". Determines the mode of
                preconditioner construction. "srht" is cheaper, requiring one pass over
                the dataset, but lower quality. "srht_2" requires two passes over the
                dataset but is better. Prefer "srht_2" unless you are running on CPU,
                in which case "srht" may be preferable.

        Returns:
            hyperparams (np.ndarray): The best hyperparams found during optimization.
            n_feval (int): The number of function evaluations during optimization.
            best_score (float): The best negative marginal log-likelihood achieved.

        Raises:
            ValueError: The input dataset is checked for validity before tuning is
                initiated. If problems are found, a ValueError will provide an
                explanation of the error.
        """
        if optim_method not in ["Powell", "Nelder-Mead"]:
            raise ValueError("optim_method must be in ['Powell', 'Nelder-Mead']")
        #If the user passed starting hyperparams, use them. Otherwise,
        #if a kernel already exists, use those hyperparameters. Otherwise...
        init_hyperparams = starting_hyperparams
        if init_hyperparams is None and self.kernel is not None:
            init_hyperparams = self.kernel.get_hyperparams()
            init_hyperparams = np.round(init_hyperparams, 1)

        if nmll_rank >= self.training_rffs:
            raise ValueError("NMLL rank must be < the number of training rffs.")
        bounds = self._run_pretuning_prep(dataset, random_seed, bounds, "approximate")
        #If kernel was only just now created by run_pretuning_prep and
        #no starting hyperparams were passed, use the mean of the bounds.
        #This is...not good in general, but running multiple restarts with NM
        #is pretty expensive, which is why user is recommended to specify a
        #starting point for NM. The mean of the bounds is a (bad) default.
        if init_hyperparams is None:
            init_hyperparams = np.mean(bounds, axis=1)

        bounds_tuples = list(map(tuple, bounds))
        if self.verbose:
            print("Now beginning NM minimization.")

        args = (dataset, nmll_rank, nmll_probes, random_seed,
                    nmll_iter, nmll_tol, pretransform_dir,
                    preconditioner_mode)

        if optim_method == "Powell":
            res = minimize(self.approximate_nmll, x0 = init_hyperparams,
                options={"maxfev":max_iter, "xtol":1e-1, "ftol":1},
                method=optim_method, args = args, bounds = bounds_tuples)
        elif optim_method == "Nelder-Mead":
            res = minimize(self.approximate_nmll, x0 = init_hyperparams,
                options={"maxfev":max_iter,
                        "xatol":1e-1, "fatol":1e-1},
                method=optim_method, args = args, bounds = bounds_tuples)

        self._post_tuning_cleanup(dataset, res.x)
        return res.x, res.nfev, res.fun



    def tune_hyperparams_crude_lbfgs(self, dataset, random_seed = 123,
            max_iter = 20, n_restarts = 1, starting_hyperparams = None,
            bounds = None, subsample = 1):
        """Tunes the hyperparameters using the L-BFGS algorithm, with
        NMLL as the objective.
        It uses either a supplied set of starting hyperparameters OR
        randomly chosen locations. If the latter, it is run
        n_restarts times. Because it uses exact NMLL rather than
        approximate, this method is only suitable to small numbers
        of random features; scaling to larger numbers of random
        features is quite poor. Using > 5000 random features in
        this method will be fairly slow. It may also be less useful
        for large datasets. This is therefore intended as a "quick-
        and-dirty" method. Often however the hyperparameters obtained
        using this method are good enough that no further fine-tuning
        is required.

        Args:
            dataset: Object of class OnlineDataset or OfflineDataset.
            random_seed (int): A random seed for the random
                number generator. Defaults to 123.
            max_iter (int): The maximum number of iterations for
                which l-bfgs should be run per restart.
            n_restarts (int): The maximum number of restarts to run l-bfgs.
            starting_hyperparams (np.ndarray): A starting point for l-bfgs
                based optimization. Defaults to None. If None, randomly
                selected locations are used.
            bounds (np.ndarray): The bounds for optimization. Defaults to
                None, in which case the kernel uses its default bounds.
                If supplied, must be a 2d numpy array of shape (num_hyperparams, 2).
            subsample (float): A value in the range [0.01,1] that indicates what
                fraction of the training set to use each time the gradient is
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
                initiated. If problems are found, a ValueError will provide an
                explanation of the error.
        """
        if subsample < 0.01 or subsample > 1:
            raise ValueError("subsample must be in the range [0.01, 1].")

        init_hyperparams = starting_hyperparams
        if init_hyperparams is None and self.kernel is not None:
            init_hyperparams = self.kernel.get_hyperparams(logspace=True)

        bounds = self._run_pretuning_prep(dataset, random_seed, bounds, "exact")
        best_x, best_score, net_iterations = None, np.inf, 0
        bounds_tuples = list(map(tuple, bounds))

        if init_hyperparams is None:
            init_hyperparams = self.kernel.get_hyperparams(logspace=True)

        rng = np.random.default_rng(random_seed)
        args, cost_fun = (dataset, subsample), self.exact_nmll_gradient

        if self.verbose:
            print("Now beginning L-BFGS minimization.")

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

            init_hyperparams = [rng.uniform(low = bounds[j,0], high = bounds[j,1])
                    for j in range(bounds.shape[0])]
            init_hyperparams = np.asarray(init_hyperparams)

        if best_x is None:
            raise ValueError("All restarts failed to find acceptable hyperparameters.")

        self._post_tuning_cleanup(dataset, best_x)
        return best_x, net_iterations, best_score
