"""Contains the tools needed to get weights for the model
using the L-SR1 optimization algorithm with an optional
but strongly recommended preconditioner as H0."""
import numpy as np
try:
    import cupy as cp
except:
    pass

from ..scoring_toolkit.exact_nmll_calcs import calc_zty



class lSR1:
    """This class contains all the tools needed to fit a model
    whose hyperparameters have already been tuned using the L-SR1
    algorithm. It is highly preferable to supply a preconditioner
    which acts as an H0 approximation, since otherwise this
    algorithm may take a long time to converge.

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
        dtype: A convenience reference to either cp.float64 or np.float64,
            depending on device.
        niter (int): The number of function evaluations performed.
        init_history_size (int): The number of previous gradient updates to store
            without any overwrite.
        recent_history_size (int): The number of previous gradient updates
            to overwrite.
        losses (list): A list of loss values. Useful for comparing rate of
            convergence with other options.
        preconditioner: Either None or a valid preconditioner object.
        stored_mvecs (ndarray): A cupy or numpy array containing the
            sk - Hk yk terms; shape (num_rffs, history_size).
        stored_nconstants (ndarray): The denominator of the Hessian
            update; shape (history_size).
    """

    def __init__(self, dataset, kernel, device, verbose, preconditioner = None,
            recent_history_size = 5):
        """Class constructor.

        Args:
        dataset: An OnlineDataset or OfflineDatset containing all the
            training data.
        kernel: A kernel object that can generate random features for
            the Dataset.
        device (str): One of 'cpu', 'gpu'. Indicates where calculations
            will be performed.
        verbose (bool): If True, print regular updates.
        preconditioner: Either None or a valid preconditioner object.
        recent_history_size (int): The number of recent gradient updates
            to store. The total number stored is this number + 50.
        """
        self.dataset = dataset
        self.kernel = kernel
        self.lambda_ = kernel.get_lambda()
        self.verbose = verbose
        self.device = device
        if device == "gpu":
            self.zero_arr = cp.zeros
            self.dtype = cp.float64
        else:
            self.zero_arr = np.zeros
            self.dtype = np.float64
        self.n_iter = 0
        self.init_history_size = 25
        self.recent_history_size = recent_history_size

        self.losses = []
        self.preconditioner = preconditioner
        self.stored_mvecs = self.zero_arr((self.kernel.get_num_rffs(),
            self.init_history_size + self.recent_history_size))
        self.stored_nconstants = self.zero_arr((self.init_history_size +
            self.recent_history_size))
        self.stored_nconstants[:] = 1


    def fit_model(self, max_iter = 500, tol = 1e-6):
        """Finds an optimal set of weights using the information already
        provided to the class constructor.

        Args:
            max_iter (int): The maximum number of iterations for L_BFGS.
            tol (float): The threshold for convergence.

        Returns:
            wvec: A cupy or numpy array depending on device that contains the
                best set of weights found. A 1d array of length self.kernel.get_num_rffs().
        """
        z_trans_y, _ = calc_zty(self.dataset, self.kernel)

        wvec = self.zero_arr((self.kernel.get_num_rffs()))
        grad = -z_trans_y.copy()
        init_norms = float((z_trans_y**2).sum())


        losses, pcounter = [1], 0
        step_size = 1

        for i in range(0, max_iter):
            shift = -step_size * self.hess_vector_prod(grad)
            wvec += shift
            grad, loss, pcounter = self.update_params(pcounter, grad,
                            shift, wvec, z_trans_y, init_norms)
            print(loss)
            losses.append(loss)
            if loss < tol:
                break

        if self.device == "gpu":
            wvec = cp.asarray(wvec)
        return wvec, self.n_iter, losses




    def update_params(self, pcounter, grad, s_k, wvec, z_trans_y, init_norms):
        """Updates the weight vector and the approximate hessian maintained
        by the algorithm.

        Args:
            pcounter (int): Which of the columns of stored_mvecs to modify.
            grad (ndarray): The previous gradient of the weights. Shape (num_rffs).
            s_k (ndarray): The amount by which the weights have shifted on
                this iteration. Shape (num_rffs).
            wvec (ndarray): The current weight values.
            z_trans_y (ndarray): The right hand side b in the equation Ax=b.
            init_norms (float): The initial norm of z_trans_y. Useful for
                ensuring the loss is scaled.

        Returns:
            wvec (ndarray): The updated wvec.
            last_wvec (ndarray): Current wvec (which is now the last wvec).
            pcounter (int): The updated pcounter.
        """
        loss, new_grad = self.cost_fun_regression(wvec, z_trans_y, init_norms)

        y_k = new_grad - grad
        y_kh = self.hess_vector_prod(y_k)
        s_kh = self.hess_vector_prod(s_k)
        denominator = (s_k - y_kh).T @ y_k
        #If the denominator of the SR1 update is too small, update the weight
        #vector using the existing Hessian and take no other action.
        update_term = y_k - s_kh
        criterion = (update_term**2).sum() * (s_k**2).sum() * 1e-8
        criterion = criterion <= np.abs(float(s_k.T @ (update_term)))
        if not criterion:
            import pdb
            pdb.set_trace()
            if self.verbose:
                print(f"Small denominator {denominator} encountered in SR1 procedure. This is a minor "
                        "nonfatal issue if infrequent and is printed only for informational "
                        "purposes.")
            return new_grad, loss, pcounter

        self.stored_mvecs[:,pcounter] = s_k - y_kh
        self.stored_nconstants[pcounter] = denominator
        pcounter += 1
        if pcounter >= (self.init_history_size + self.recent_history_size):
            pcounter = self.init_history_size
        return new_grad, loss, pcounter



    def hess_vector_prod(self, ivec):
        """Takes the product of the approximate Hessian with an input vector.

        Args:
            ivec (ndarray): A cupy or numpy array of shape (num_rffs) with which
                to take the product.

        Returns:
            ovec (ndarray): Hk @ ivec.
        """
        ovec = (self.stored_mvecs * ivec[:,None]).sum(axis=0) / self.stored_nconstants
        ovec = (ovec[None,:] * self.stored_mvecs).sum(axis=1)
        if self.preconditioner is not None:
            ovec += self.preconditioner.batch_matvec(ivec[:,None])[:,0]
        return ovec



    def cost_fun_regression(self, wvec, z_trans_y, init_norm):
        """The cost function for finding the weights for
        regression. Returns both the current loss and the gradient.

        Args:
            wvec (np.ndarray): The current set of weights.
            z_trans_y: A cupy or numpy array (depending on device)
                containing Z.T @ y, where Z is the random features
                generated for all of the training datapoints.
            init_norm (float): The initial loss. Used to scale the error
                so it is on a 0-1 scale.

        Returns:
            loss (float): The current loss.
            grad (np.ndarray): The gradient for the current set of weights.
        """
        xprod = self.lambda_**2 * wvec
        if self.dataset.pretransformed:
            for xdata in self.dataset.get_chunked_x_data():
                xprod += (xdata.T @ (xdata @ wvec))
        else:
            for xdata in self.dataset.get_chunked_x_data():
                xtrans = self.kernel.transform_x(xdata)
                xprod += (xtrans.T @ (xtrans @ wvec))


        grad = xprod - z_trans_y
        loss = float((grad**2).sum()) / init_norm

        if self.verbose:
            if self.n_iter % 5 == 0:
                print(f"Nfev {self.n_iter} complete")
        self.n_iter += 1
        return loss, grad
