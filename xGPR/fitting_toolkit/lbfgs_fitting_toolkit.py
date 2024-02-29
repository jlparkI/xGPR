"""Contains the tools needed to get weights for the model
using the L-BFGS optimization algorithm."""
import numpy as np
try:
    import cupy as cp
except:
    pass
from scipy.optimize import minimize

from ..scoring_toolkit.exact_nmll_calcs import calc_zty



class lBFGSModelFit:
    """This class contains all the tools needed to fit a model
    whose hyperparameters have already been tuned using the L-BFGS
    algorithm. This is usually slower than preconditioned-CG
    but for small datasets may be a good option.

    Attributes:
        dataset: An OnlineDataset or OfflineDatset containing all the
            training data.
        kernel: A kernel object that can generate random features for
            the Dataset.
        lambda_ (float): The noise hyperparameter shared across all kernels.
        verbose (bool): If True, print regular updates.
        device (str): One of 'cpu', 'gpu'. Indicates where calculations
            will be performed.
        zero_arr: A convenience reference to either cp.zeros or np.zeros.
        log: A convenience reference to either cp.log or np.log.
        dtype: A convenience reference to either cp.float64 or np.float64,
            depending on device.
        niter (int): The number of function evaluations performed.
        task_type (str): One of "regression", "classification", "discriminant".
        n_classes (int): The number of classes if performing classification;
            ignored otherwise.
    """

    def __init__(self, dataset, kernel, device, verbose, task_type = "regression",
            n_classes = 2):
        """Class constructor.

        Args:
            dataset: An OnlineDataset or OfflineDatset containing all the
                training data.
            kernel: A kernel object that can generate random features for
                the Dataset.
            device (str): One of 'cpu', 'gpu'. Indicates where calculations
                will be performed.
            verbose (bool): If True, print regular updates.
            task_type (str): One of "regression", "classification", "discriminant".
            n_classes (int): The number of categories if performing classification;
                ignored otherwise.
        """
        self.dataset = dataset
        self.kernel = kernel
        self.lambda_ = kernel.get_lambda()
        self.verbose = verbose
        self.device = device
        if device == "gpu":
            self.zero_arr = cp.zeros
            self.dtype = cp.float64
            self.log = cp.log
        else:
            self.zero_arr = np.zeros
            self.dtype = np.float64
            self.log = np.log
        self.n_iter = 0

        self.task_type = task_type
        self.n_classes = n_classes




    def fit_model_lbfgs(self, max_iter = 500, tol = 3e-09,
            x_mean = None, targets = None):
        """Finds an optimal set of weights using the information already
        provided to the class constructor.

        Args:
            max_iter (int): The maximum number of iterations for L_BFGS.
            tol (float): The threshold for convergence. User not currently
                allowed to specify since setting a larger / smaller tol
                can result in very poor results with L-BFGS (either very
                large number of iterations or poor performance).
            x_mean: Either None or an ndarray of shape (num_rffs) storing the mean
                of the data. Used for discriminant-type classifiers only.
            targets: Either None or an ndarray of shape (num_rffs, n_classes).
                Used for discriminant type classifiers only.

        Returns:
            wvec: A cupy or numpy array depending on device that contains the
                best set of weights found. A 1d array of length self.kernel.get_num_rffs().
        """
        if self.task_type == "regression":
            z_trans_y, _ = calc_zty(self.dataset, self.kernel)
            init_weights = np.zeros((self.kernel.get_num_rffs()))
            res = minimize(self.regression_cost_fun,
                    options={"maxiter":max_iter, "ftol":tol},
                    method = "L-BFGS-B",
                    x0 = init_weights, args = (z_trans_y),
                    jac = True, bounds = None)

        elif self.task_type == "discriminant":
            if x_mean is None or targets is None:
                raise ValueError("x_mean and targets must be supplied for "
                        "fitting a discriminant.")

            init_weights = np.zeros((self.kernel.get_num_rffs() * self.n_classes))
            res = minimize(self.discriminant_cost_fun,
                    options={"maxiter":max_iter, "ftol":tol},
                    method = "L-BFGS-B", args = (x_mean, targets),
                    x0 = init_weights, jac = True, bounds = None)


        elif self.task_type == "classification" and self.n_classes == 2:
            init_weights = np.zeros((self.kernel.get_num_rffs()))
            res = minimize(self.binary_classification_cost_fun,
                    options={"maxiter":max_iter, "ftol":tol},
                    method = "L-BFGS-B",
                    x0 = init_weights, jac = True, bounds = None)
        else:
            raise ValueError("Multiclass kernel logistic regression has not "
                    "been implemented yet.")


        wvec = res.x
        if self.device == "gpu":
            wvec = cp.asarray(wvec)
        if self.task_type == "discriminant":
            wvec = wvec.reshape((self.kernel.get_num_rffs(), self.n_classes))
        return wvec, self.n_iter, []


    def regression_cost_fun(self, weights, z_trans_y):
        """The cost function for regression.

        Args:
            weights (np.ndarray): The current set of weights.
            z_trans_y: A cupy or numpy array (depending on device)
                containing Z.T @ y, where Z is the random features
                generated for all of the training datapoints.

        Returns:
            loss (float): The current loss.
            grad (np.ndarray): The gradient for the current set of weights.
        """
        wvec = weights
        if self.device == "gpu":
            wvec = cp.asarray(wvec).astype(self.dtype)
        xprod = self.lambda_**2 * wvec
        for (xdata, ldata) in self.dataset.get_chunked_x_data():
            xtrans = self.kernel.transform_x(xdata, ldata)
            xprod += (xtrans.T @ (xtrans @ wvec))


        grad = xprod - z_trans_y
        loss = 0.5 * (wvec.T @ xprod) - z_trans_y.T @ wvec

        if self.device == "gpu":
            grad = cp.asnumpy(grad).astype(np.float64)
        if self.verbose:
            if self.n_iter % 5 == 0:
                print(f"Nfev {self.n_iter} complete")
        self.n_iter += 1
        return float(loss), grad


    def discriminant_cost_fun(self, weights, x_mean, targets):
        """The cost function for regression.

        Args:
            weights (np.ndarray): The current set of weights.
            z_trans_y: A cupy or numpy array (depending on device)
                containing Z.T @ y, where Z is the random features
                generated for all of the training datapoints.

        Returns:
            loss (float): The current loss.
            grad (np.ndarray): The gradient for the current set of weights.
        """
        wvec = weights.reshape((self.kernel.get_num_rffs(), self.n_classes))
        if self.device == "gpu":
            wvec = cp.asarray(wvec).astype(self.dtype)
            xprod = cp.zeros((self.kernel.get_num_rffs(), self.n_classes))
        else:
            xprod = np.zeros((self.kernel.get_num_rffs(), self.n_classes))

        for (xdata, ldata) in self.dataset.get_chunked_x_data():
            xtrans = self.kernel.transform_x(xdata, ldata) - x_mean[None,:]
            xprod += (xtrans.T @ (xtrans @ wvec))

        xprod /= self.dataset.get_ndatapoints()
        xprod += self.lambda_**2 * wvec

        grad = xprod - targets
        loss = 0.5 * (wvec * xprod).sum() - (targets * wvec).sum()

        if self.device == "gpu":
            grad = cp.asnumpy(grad).astype(np.float64)
        if self.verbose:
            if self.n_iter % 5 == 0:
                print(f"Nfev {self.n_iter} complete")
        self.n_iter += 1
        return float(loss), grad.reshape((self.kernel.get_num_rffs() * self.n_classes))



    def binary_classification_cost_fun(self, weights):
        """The cost function for binary classification.

        Args:
            weights (np.ndarray): The current set of weights.

        Returns:
            loss (float): The current loss.
            grad (np.ndarray): The gradient for the current set of weights.
        """
        wvec = weights
        if self.device == "gpu":
            wvec = cp.asarray(wvec).astype(self.dtype)
        grad = self.lambda_**2 * wvec
        loss = self.lambda_**2 * (wvec.T @ wvec)

        for xdata, ydata, ldata in self.dataset.get_chunked_data():
            xtrans = self.kernel.transform_x(xdata, ldata)
            pred = (xtrans @ -wvec).clip(max=30)
            pred = (1 / (1 + 2.71828**pred)).flatten()
            pred = pred.clip(min=1e-12, max=1e12)
            grad += (xtrans.T @ (pred - ydata)).flatten()
            loss -= (self.log(pred) * ydata + (1 - ydata) * self.log(1 - pred)).sum()


        if self.device == "gpu":
            grad = cp.asnumpy(grad).astype(np.float64)
        if self.verbose:
            if self.n_iter % 5 == 0:
                print(f"Nfev {self.n_iter} complete")
        self.n_iter += 1
        return float(loss), grad


    def multiclass_classification_cost_fun(self, weights, z_trans_y):
        """The cost function for multiclass classification.

        Args:
            weights (np.ndarray): The current set of weights.
            z_trans_y: A cupy or numpy array (depending on device)
                containing Z.T @ y, where Z is the random features
                generated for all of the training datapoints.

        Returns:
            loss (float): The current loss.
            grad (np.ndarray): The gradient for the current set of weights.
        """
        wvec = weights
        if self.device == "gpu":
            wvec = cp.asarray(wvec).astype(self.dtype)
        xprod = self.lambda_**2 * wvec
        for (xdata, ldata) in self.dataset.get_chunked_x_data():
            xtrans = self.kernel.transform_x(xdata, ldata)
            xprod += (xtrans.T @ (xtrans @ wvec))


        grad = xprod - z_trans_y
        loss = 0.5 * (wvec.T @ xprod) - z_trans_y.T @ wvec


        if self.device == "gpu":
            grad = cp.asnumpy(grad).astype(np.float64)
        if self.verbose:
            if self.n_iter % 5 == 0:
                print(f"Nfev {self.n_iter} complete")
        self.n_iter += 1
        return float(loss), grad
