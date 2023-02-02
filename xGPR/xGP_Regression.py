"""Describes the xGPRegression class.

The xGPRegression class provides the tools needed to fit a regression
model and make predictions for new datapoints. It inherits from
GPModelBaseclass.
"""
import warnings

try:
    import cupy as cp
    import cupyx as cpx
    from cupyx.scipy.sparse.linalg import cg as Cuda_CG
except:
    print("CuPy not detected. xGPR will run in CPU-only mode.")
import numpy as np
from scipy.linalg import cho_solve
from scipy.sparse.linalg import cg as CPU_CG

from .constants import constants
from .regression_baseclass import GPRegressionBaseclass
from .preconditioners.rand_nys_preconditioners import Cuda_RandNysPreconditioner
from .preconditioners.rand_nys_preconditioners import CPU_RandNysPreconditioner
from .preconditioners.tuning_preconditioners import RandNysTuningPreconditioner
from .fitting_toolkit.lbfgs_fitting_toolkit import lBFGSModelFit
from .fitting_toolkit.sgd_fitting_toolkit import sgdModelFit
from .fitting_toolkit.ams_grad_toolkit import amsModelFit

from .scoring_tools.approximate_nmll_calcs import estimate_logdet, estimate_nmll
from .scoring_tools.gradient_tools import exact_nmll_reg_grad
from .scoring_tools.gradient_tools import minibatch_reg_grad
from .scoring_tools.probe_generators import generate_normal_probes_gpu
from .scoring_tools.probe_generators import generate_normal_probes_cpu

from cg_tools import CPU_ConjugateGrad, GPU_ConjugateGrad
from .cg_toolkit.cg_linear_operators import Cuda_CGLinearOperator, CPU_CGLinearOperator


class xGPRegression(GPRegressionBaseclass):
    """A subclass of GPRegressionBaseclass that houses methods
    unique to regression problems. It does not have
    any attributes unique to it aside from those
    of the parent class."""

    def __init__(self, training_rffs,
                    fitting_rffs,
                    variance_rffs = 16,
                    kernel_choice="rbf",
                    device = "cpu",
                    kernel_specific_params = constants.DEFAULT_KERNEL_SPEC_PARMS,
                    verbose = True,
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
            double_precision_fht (bool): If False, use single precision floats to generate
                random features. This can increase speed but may result in a slight (usually
                negligible) loss of accuracy.
        """
        super().__init__(training_rffs, fitting_rffs, variance_rffs,
                        kernel_choice, device, kernel_specific_params,
                        verbose, double_precision_fht)

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
                xfeatures = xfeatures[:,:self.variance_rffs]
                pred_var = (self.var @ xfeatures.T).T
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


    def transform_data(self, input_x, chunk_size = 2000):
        """Generate the random features for each chunk
        of an input array. This function is a generator
        so it will yield the random features as blocks
        of shape (chunk_size, fitting_rffs).

        Args:
            input_x (np.ndarray): The input data. Should be a 2d numpy
                array (if non-convolution kernel) or 3d (if convolution
                kernel).
            chunk_size (int): The number of datapoints to process at
                a time. Lower values limit memory consumption. Defaults
                to 2000.

        Yields:
            x_trans (array): An array containing the random features
                generated for a chunk of the input. Shape is
                (chunk_size, fitting_rffs).

        Raises:
            ValueError: If the dimensionality or type of the input does
                not match what is expected, or if the model has
                not yet been fitted, a ValueError is raised.
        """
        xdata = self.pre_prediction_checks(input_x, False)
        for i in range(0, xdata.shape[0], chunk_size):
            cutoff = min(i + chunk_size, xdata.shape[0])
            yield self.kernel.transform_x(xdata[i:cutoff, :])



    def pre_prediction_checks(self, input_x, get_var):
        """Checks the input data to predict_scores to ensure validity.

        Args:
            input_x (np.ndarray): A numpy array containing the input data.
            get_var (bool): Whether a variance calculation is desired.

        Returns:
            x_array: A cupy array (if self.device is gpu) or a reference
                to the unmodified input array otherwise.

        Raises:
            ValueError: If invalid inputs are supplied,
                a detailed ValueError is raised to explain.
        """
        x_array = input_x
        if self.kernel is None:
            raise ValueError("Model has not yet been successfully fitted.")
        if not self.kernel.validate_new_datapoints(input_x):
            raise ValueError("The input has incorrect dimensionality.")
        #TODO: Add proper variance calc for linear kernel.
        if self.var is None and get_var:
            raise ValueError("Variance was requested but suppress_var "
                    "was selected when fitting or a linear kernel was "
                    "used, meaning that variance has not been generated.")
        if self.device == "gpu":
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
            x_array = cp.asarray(input_x)

        return x_array


    def get_hyperparams(self):
        """Simple helper function to return hyperparameters if the model
        has already been tuned or fitted."""
        if self.kernel is None:
            return None
        return self.kernel.get_hyperparams()


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



    def _calc_zTy(self, dataset):
        """Calculates the vector Z^T y. This function
        should never be called if a kernel has not
        already been initialized.

        Args:
            dataset: An Dataset object that can supply
                chunked data.

        Returns:
            z_trans_y (array): A shape (num_rffs)
                array that contains Z^T y.
            y_trans_y (float): The value y^T y.
        """
        if self.device == "gpu":
            z_trans_y = cp.zeros((self.kernel.get_num_rffs()))
        else:
            z_trans_y = np.zeros((self.kernel.get_num_rffs()))

        y_trans_y = 0

        if dataset.pretransformed:
            for xdata, ydata in dataset.get_chunked_data():
                z_trans_y += xdata.T @ ydata
                y_trans_y += float( (ydata**2).sum() )
        else:
            for xdata, ydata in dataset.get_chunked_data():
                zdata = self.kernel.transform_x(xdata)
                z_trans_y += zdata.T @ ydata
                y_trans_y += float( (ydata**2).sum() )
        return z_trans_y, y_trans_y


    def _calc_weights_exact(self, dataset):
        """Calculates the weights when fitting the model using
        matrix decomposition. Exact and fast for small numbers
        of random features but poor scaling.

        Args:
            dataset: Either OnlineDataset or OfflineDataset,
                containing the information on the dataset we
                are fitting.

        Returns:
            weights: A cupy or numpy array of shape (M) for M
                random features.
        """
        z_trans_z, z_trans_y, _ = self._calc_design_mat(dataset)
        lambda_p = self.kernel.get_hyperparams(logspace=False)[0]
        z_trans_z.flat[::z_trans_z.shape[0]+1] += lambda_p**2
        _, weights = self._direct_weight_calc(z_trans_z, z_trans_y)
        return weights


    def _calc_weights_cg(self, dataset, cg_tol = 1e-4, max_iter = 500,
                        preconditioner = None):
        """Calculates the weights when fitting the model using
        preconditioned CG. Good scaling but slower for small
        numbers of random features.

        Args:
            dataset: Either OnlineDataset or OfflineDataset.
            cg_tol (float): The threshold below which cg is deemed to have
                converged. Defaults to 1e-5.
            max_iter (int): The maximum number of iterations before
                CG is deemed to have failed to converge.
            preconditioner: Either None or a valid Preconditioner (e.g.
                CudaRandomizedPreconditioner, CPURandomizedPreconditioner
                etc). If None, no preconditioning is used. Otherwise,
                the preconditioner is used for CG. The preconditioner
                can be built by calling self.build_preconditioner
                with appropriate arguments.
            starting_guess: Either None or a cupy or numpy array containing
                starting guess values for the weights. Defaults to None.

        Returns:
            weights: A cupy or numpy array of shape (M) for M
                random features.
            n_iter (int): The number of CG iterations.
            losses (list): The loss on each iteration; for diagnostic
                purposes.
        """
        if self.device == "gpu":
            cg_operator = GPU_ConjugateGrad()
            resid = cp.zeros((self.kernel.get_num_rffs(), 2, 1))
        else:
            cg_operator = CPU_ConjugateGrad()
            resid = np.zeros((self.kernel.get_num_rffs(), 2, 1))

        z_trans_y, _ = self._calc_zTy(dataset)
        resid[:,0,:] = z_trans_y[:,None] / dataset.get_ndatapoints()

        weights, converged, n_iter, losses = cg_operator.fit(dataset, self.kernel,
                preconditioner, resid, max_iter, cg_tol, self.verbose,
                nmll_settings = False)
        weights *= dataset.get_ndatapoints()
        if not converged:
            warnings.warn("Conjugate gradients failed to converge! Try refitting "
                        "the model with updated settings.")

        if self.verbose:
            print(f"CG iterations: {n_iter}")
        return weights, n_iter, losses


    def _calc_weights_cg_lib_ext(self, dataset, cg_tol = 1e-5, max_iter = 500,
                        preconditioner = None):
        """Calculates the weights when fitting the model using
        preconditioned CG. Good scaling but slower for small
        numbers of random features. Uses the CG implementation
        in Scipy and Cupy instead of the internal implementation
        (we've found these to provide very similar results,
        but it is good to be able to use either, also the
        internal implementation can keep track of loss values
        for diagnostics.)

        Args:
            dataset: Either OnlineDataset or OfflineDataset.
            cg_tol (float): The threshold below which cg is deemed to have
                converged. Defaults to 1e-5.
            max_iter (int): The maximum number of iterations before
                CG is deemed to have failed to converge.
            preconditioner: Either None or a valid Preconditioner (e.g.
                CudaRandomizedPreconditioner, CPURandomizedPreconditioner
                etc). If None, no preconditioning is used. Otherwise,
                the preconditioner is used for CG. The preconditioner
                can be built by calling self.build_preconditioner
                with appropriate arguments.
            starting_guess: Either None or a cupy or numpy array containing
                starting guess values for the weights. Defaults to None.

        Returns:
            weights: A cupy or numpy array of shape (M) for M
                random features.
            n_iter (int): The number of CG iterations.
            losses (list): The loss on each iteration; for diagnostic
                purposes.
        """
        if self.device == "gpu":
            cg_fun = Cuda_CG
            cg_operator = Cuda_CGLinearOperator(dataset, self.kernel,
                    self.verbose)
        else:
            cg_fun = CPU_CG
            cg_operator = CPU_CGLinearOperator(dataset, self.kernel,
                    self.verbose)

        z_trans_y, _ = self._calc_zTy(dataset)

        weights, convergence = cg_fun(A = cg_operator, b = z_trans_y,
                M = preconditioner, tol = cg_tol, atol=0)

        if convergence != 0:
            warnings.warn("Conjugate gradients failed to converge! Try refitting "
                        "the model with updated settings.")

        return weights, cg_operator.n_iter, []


    def _calc_weights_lbfgs(self, fitting_dataset, tol, max_iter = 500):
        """Calculates the weights when fitting the model using
        L-BFGS. Only recommended for small datasets.

        Args:
            fitting_dataset: An OnlineDataset or OfflineDataset with
                the training data to be fitted.
            tol (float): The threshold for convergence.
            max_iter (int): The maximum number of L-BFGS iterations.
            random_state (int): Seed for the random number generator.

        Returns:
            weights: A cupy or numpy array (depending on self.device)
                containing the resulting weights from fitting.
            niter (int): The number of function evaluations required
                to obtain this set of weights.
        """
        model_fitter = lBFGSModelFit(fitting_dataset, self.kernel,
                    self.device, self.verbose)
        weights = model_fitter.fit_model_lbfgs(max_iter, tol)
        n_iter = model_fitter.get_niter()
        return weights, n_iter




    def _calc_weights_sgd(self, fitting_dataset, tol, max_iter = 50,
            preconditioner = None, manual_lr = None):
        """Calculates the weights when fitting the model using
        stochastic variance reduction gradient descent, preferably
        with preconditioning (although can also function without).
        Excellent scaling, but a little less accurate than CG.

        Args:
            fitting_dataset: An OnlineDataset or OfflineDataset with
                the training data to be fitted.
            tol (float): The threshold for convergence.
            max_iter (int): The maximum number of epochs.
            preconditioner: Either None or a valid Preconditioner object.
            manual_lr (float): Either None or a float. If not None, this
                is a user-specified initial learning rate. If None, find
                a good initial learning rate using autotuning.

        Returns:
            weights: A cupy or numpy array (depending on self.device)
                containing the resulting weights from fitting.
            niter (int): The number of function evaluations required
                to obtain this set of weights.
            losses (list): A list of losses at each iteration.
        """
        model_fitter = sgdModelFit(self.kernel.get_lambda(), self.device, self.verbose)
        weights, losses = model_fitter.fit_model(fitting_dataset,
                    self.kernel, tol = tol, max_epochs = max_iter,
                    preconditioner = preconditioner, manual_lr = manual_lr)
        n_iter = model_fitter.get_niter()
        return weights, n_iter, losses


    def _calc_weights_ams(self, fitting_dataset, tol, max_iter = 50):
        """Calculates the weights when fitting the model using
        AMSGrad. Use for testing purposes only -- this is not currently
        competitive with our other methods on tough problems.

        Args:
            fitting_dataset: An OnlineDataset or OfflineDataset with
                the training data to be fitted.
            tol (float): The threshold for convergence.
            max_iter (int): The maximum number of epochs.

        Returns:
            weights: A cupy or numpy array (depending on self.device)
                containing the resulting weights from fitting.
            niter (int): The number of function evaluations required
                to obtain this set of weights.
        """
        model_fitter = amsModelFit(self.kernel.get_lambda(), self.device, self.verbose)
        weights, losses = model_fitter.fit_model(fitting_dataset,
                    self.kernel, tol = tol, max_epochs = max_iter)
        n_iter = model_fitter.get_niter()
        return weights, n_iter, losses



    def _calc_variance(self, dataset):
        """Calculates the var matrix used for calculating
        posterior predictive variance on new datapoints. We
        only ever use closed-form matrix-decomposition based
        calculations here since the variance does not need to
        be approximated as accurately as the posterior predictive
        mean, so we can restrict the user to a smaller number of
        random features (defined in constants.constants).

        Args:
            dataset: Either an OnlineDataset or an OfflineDataset containing
                the data that needs to be fitted.

        Returns:
            var: A cupy or numpy array of shape (M, M) where M is the
                number of random features.
        """
        if self.verbose:
            print("Estimating variance...")
        #This is a very naughty hack.
        #TODO: Add a proper variance calc for linear.
        if self.kernel_choice == "Linear":
            return None
        z_trans_z = self._calc_var_design_mat(dataset)
        lambda_ = self.kernel.get_lambda()
        z_trans_z.flat[::z_trans_z.shape[0]+1] += lambda_**2
        if self.device == "cpu":
            var = np.linalg.pinv(z_trans_z)
        else:
            var = cp.linalg.pinv(z_trans_z)
        if self.verbose:
            print("Variance estimated.")
        return var


    def _direct_weight_calc(self, chol_z_trans_z, z_trans_y):
        """Calculates the cholesky decomposition of (z^T z + lambda)^-1
        and then uses this to calculate the weights as (z^T z + lambda)^-1 z^T y.
        This exact calculation is only suitable for < 10,000 random features or so;
        cholesky has O(M^3) scaling.

        Args:
            z_trans_z: An M x M cupy or numpy matrix where M is the number of
                random features formed from z^T z when z is the random features
                generated for raw input data X.
            z_trans_y: A length M cupy or numpy array where M is the number of
                random features, formed from z^T y.

        Returns:
            chol_z_trans_z: The cholesky decomposition of z_trans_z. An M x M
                cupy or numpy matrix.
            weights: A length M cupy or numpy array containing the weights.
        """
        lambda_p = self.kernel.get_hyperparams(logspace=False)[0]
        chol_z_trans_z.flat[::chol_z_trans_z.shape[0]+1] += lambda_p**2
        if self.device == "cpu":
            chol_z_trans_z = np.linalg.cholesky(chol_z_trans_z)
            weights = cho_solve((chol_z_trans_z, True), z_trans_y)
        else:
            chol_z_trans_z = cp.linalg.cholesky(chol_z_trans_z)
            weights = cpx.scipy.linalg.solve_triangular(chol_z_trans_z,
                            z_trans_y, lower=True)
            weights = cpx.scipy.linalg.solve_triangular(chol_z_trans_z.T,
                            weights, lower=False)
        return chol_z_trans_z, weights



    def _calc_design_mat(self, dataset):
        """Calculates the z_trans_z (z^T z) matrix where Z is the random
        features generated from raw input data X. Also generates
        y^T y and z^T y.

        Args:
            dataset: An OnlineDataset or OfflineDataset object storing
                the data we will use.

        Returns:
            z_trans_z: The cupy or numpy matrix resulting from z^T z. Will
                be shape M x M for M random features.
            z_trans_y: The cupy or numpy length M array resulting from
                z^T y for M random features.
            y_trans_y (float): The result of the dot product of y with itself.
                Used for some marginal likelihood calculations.
        """
        num_rffs = self.kernel.get_num_rffs()
        if self.device == "cpu":
            z_trans_z, z_trans_y = np.zeros((num_rffs, num_rffs)), np.zeros((num_rffs))
        else:
            z_trans_z = cp.zeros((num_rffs, num_rffs))
            z_trans_y = cp.zeros((num_rffs))
        y_trans_y = 0.0
        if dataset.pretransformed:
            for xfeatures, ydata in dataset.get_chunked_data():
                z_trans_y += xfeatures.T @ ydata
                z_trans_z += xfeatures.T @ xfeatures
                y_trans_y += ydata.T @ ydata
        else:
            for xdata, ydata in dataset.get_chunked_data():
                xfeatures = self.kernel.transform_x(xdata)
                z_trans_y += xfeatures.T @ ydata
                z_trans_z += xfeatures.T @ xfeatures
                y_trans_y += ydata.T @ ydata
        return z_trans_z, z_trans_y, float(y_trans_y)


    def _calc_var_design_mat(self, dataset):
        """Calculates the z_trans_z (z^T z) matrix where Z is the random
        features generated from raw input data X, for calculating
        variance only (since in this case we only use up to
        self.variance_rffs of the features generated).

        Args:
            dataset: An OnlineDataset or OfflineDataset object storing
                the data we will use.

        Returns:
            z_trans_z: The cupy or numpy matrix resulting from z^T z. Will
                be shape M x M for M random features.
        """
        num_rffs = self.variance_rffs
        if self.device == "cpu":
            z_trans_z = np.zeros((num_rffs, num_rffs))
        else:
            z_trans_z = cp.zeros((num_rffs, num_rffs))
        if dataset.pretransformed:
            for xfeatures in dataset.get_chunked_x_data():
                z_trans_z += xfeatures[:,:num_rffs].T @ xfeatures[:,:num_rffs]
        else:
            for xdata in dataset.get_chunked_x_data():
                xfeatures = self.kernel.transform_x(xdata)
                z_trans_z += xfeatures[:,:num_rffs].T @ xfeatures[:,:num_rffs]
        return z_trans_z


    def calc_gradient_terms(self, dataset, subsample = 1):
        """Calculates terms needed for the gradient calculation.

        Args:
            dataset: An OnlineDataset or OfflineDataset with the
                raw data we need for these calculations.
            subsample (float): A value in the range [0.01,1] that indicates what
                fraction of the training set to use each time the gradient is
                calculated (the same subset is used every time). In general, 1
                will give better results, but using a subsampled subset can be
                a fast way to find the (approximate) location of a good
                hyperparameter set.

        Returns:
            z_trans_z: The M x M cupy or numpy matrix for M random features.
            z_trans_y: The length M cupy or numpy array for M random features.
            y_trans_y (float): The dot product of y with itself.
            dz_dsigma_ty (array): Derivative w/r/t kernel-specific hyperparams times y.
            inner_deriv (array): Derivative for the log determinant portion of the NMLL.
            ndatapoints (int): The number of datapoints.
        """
        if subsample > 1 or subsample < 0.01:
            raise ValueError("Subsample must be in the range [0.01, 1].")

        num_rffs = self.kernel.get_num_rffs()
        hparams = self.kernel.get_hyperparams()
        if self.device == "cpu":
            z_trans_z = np.zeros((num_rffs, num_rffs))
            z_trans_y = np.zeros((num_rffs))
            dz_dsigma_ty = np.zeros((num_rffs, hparams.shape[0] - 2))
            inner_deriv = np.zeros((num_rffs, num_rffs,
                                    hparams.shape[0] - 2))
            transpose = np.transpose
        else:
            z_trans_z = cp.zeros((num_rffs, num_rffs))
            z_trans_y = cp.zeros((num_rffs))
            dz_dsigma_ty = cp.zeros((num_rffs, hparams.shape[0] - 2))
            inner_deriv = cp.zeros((num_rffs, num_rffs,
                                    hparams.shape[0] - 2))
            transpose = cp.transpose

        y_trans_y = 0
        ndatapoints = 0

        if subsample == 1:
            for xdata, ydata in dataset.get_chunked_data():
                xfeatures, dz_dsigma = self.kernel.kernel_specific_gradient(xdata)
                z_trans_y += xfeatures.T @ ydata
                z_trans_z += xfeatures.T @ xfeatures
                y_trans_y += ydata.T @ ydata
                ndatapoints += xfeatures.shape[0]

                for i in range(dz_dsigma.shape[2]):
                    dz_dsigma_ty[:,i] += dz_dsigma[:,:,i].T @ ydata
                    inner_deriv[:,:,i] += dz_dsigma[:,:,i].T @ xfeatures
        else:
            rng = np.random.default_rng(123)
            for xdata, ydata in dataset.get_chunked_data():
                idx_size = max(1, int(subsample * xdata.shape[0]))
                idx = rng.choice(xdata.shape[0], idx_size, replace=False)
                xdata, ydata = xdata[idx,...], ydata[idx]
                xfeatures, dz_dsigma = self.kernel.kernel_specific_gradient(xdata)
                z_trans_y += xfeatures.T @ ydata
                z_trans_z += xfeatures.T @ xfeatures
                y_trans_y += ydata.T @ ydata
                ndatapoints += xdata.shape[0]

                for i in range(dz_dsigma.shape[2]):
                    dz_dsigma_ty[:,i] += dz_dsigma[:,:,i].T @ ydata
                    inner_deriv[:,:,i] += dz_dsigma[:,:,i].T @ xfeatures

        inner_deriv += transpose(inner_deriv, (1,0,2))
        return z_trans_z, z_trans_y, float(y_trans_y), dz_dsigma_ty, inner_deriv, ndatapoints






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
        z_trans_z, z_trans_y, y_trans_y = self._calc_design_mat(dataset)
        if self.device == "cpu":
            diagfunc, logfunc = np.diag, np.log
        else:
            diagfunc, logfunc = cp.diag, cp.log

        lambda_p = self.kernel.get_lambda()
        #Direct weight calculation may fail IF the hyperparameters supplied
        #lead to a singular design matrix. This is rare, but be prepared to
        #handle if this problem is encountered.
        try:
            chol_z_trans_z, weights = self._direct_weight_calc(z_trans_z, z_trans_y)
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
                        self.calc_gradient_terms(dataset, subsample)
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
            z_trans_y, y_trans_y = self._calc_zTy(train_dataset)
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
                        preconditioner)
        nmll = estimate_nmll(train_dataset, self.kernel, logdet, x_k,
                        z_trans_y, y_trans_y)
        if self.verbose:
            print("NMLL evaluation completed.")
        if pretransform_dir is not None:
            train_dataset.delete_dataset_files()
        return float(nmll)


    def minibatch_nmll_gradient(self, params, xdata, ydata):
        """Calculates the gradient of the negative marginal log likelihood w/r/t
        the hyperparameters for a specified set of hyperparameters using
        exact methods (matrix decompositions), for a MINIBATCH ONLY.
        The minibatch should contain less than 3,000 datapoints. Given this,
        the gradient can be calculated very efficiently using exact methods.
        Since this uses a Cholesky decomposition, it may exhibit numerical
        instability for extreme hyperparameter values.

        Args:
            params (np.ndarray): The set of hyperparameters at which
                to calculate the gradient.
            xdata (array): A cupy or numpy array of shape (ndatapoints, nfeatures)
                containing the raw data for the minibatch.
            ydata (array): A cupy or numpy arrayof shape (ndatapoints) containing
                the minibatch raw y data.

        Returns:
            grad (np.ndarray): The gradient of the NMLL w/r/t the hyperparameters.
        """
        self.kernel.set_hyperparams(params, logspace=True)
        hparams = self.kernel.get_hyperparams(logspace=False)

        xdata, dz_dsigma = self.kernel.kernel_specific_gradient(xdata)
        grad = minibatch_reg_grad(xdata, ydata, dz_dsigma, hparams, self.device)

        return grad
