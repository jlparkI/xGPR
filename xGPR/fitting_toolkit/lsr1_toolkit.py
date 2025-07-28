"""Contains the tools needed to get weights for the model
using the L-SR1 optimization algorithm with an optional
but strongly recommended preconditioner as H0."""
import numpy as np
try:
    import cupy as cp
except:
    pass

from ..scoring_toolkit.exact_nmll_calcs import calc_zty



class lSR1_classification:
    """This class contains all the tools needed to fit a model
    whose hyperparameters have already been tuned using the L-SR1
    algorithm. It is highly preferable to supply a preconditioner
    which acts as an H0 approximation, since otherwise this
    algorithm may take a long time to converge. It is intended
    for use with classification (CG is preferable for regression).

    Attributes:
        dataset: An OnlineDataset or OfflineDatset containing all the
            training data.
        kernel: A kernel object that can generate random features for
            the Dataset.
        lambda_ (float): The noise hyperparameter shared across all kernels.
        verbose (bool): If True, print regular updates.
        device (str): One of 'cpu', 'cuda'. Indicates where calculations
            will be performed.
        zero_arr: A convenience reference to either cp.zeros or np.zeros.
        dtype: A convenience reference to either cp.float64 or np.float64,
            depending on device.
        n_iter (int): The number of function evaluations performed.
        n_updates (int): The number of Hessian updates to date. Not necessarily
            the same as n_iter, since Hessian updates can be skipped.
        history_size (int): The number of previous gradient updates to store.
        losses (list): A list of loss values. Useful for comparing rate of
            convergence with other options.
        preconditioner: Either None or a valid preconditioner object.
        stored_mvecs (ndarray): A cupy or numpy array containing the
            sk - Hk yk terms; shape (num_rffs, history_size).
        stored_nconstants (ndarray): The denominator of the Hessian
            update; shape (history_size).
    """

    def __init__(self, dataset, kernel, device, verbose,
            preconditioner = None, history_size = 50):
        """Class constructor.

        Args:
            dataset: An OnlineDataset or OfflineDatset containing all the
                training data.
            kernel: A kernel object that can generate random features for
                the Dataset.
            device (str): One of 'cpu', 'cuda'. Indicates where calculations
                will be performed.
            verbose (bool): If True, print regular updates.
            preconditioner: Either None or a valid preconditioner object.
            history_size (int): The number of recent gradient updates
                to store.
        """
        self.dataset = dataset
        self.kernel = kernel
        self.lambda_ = kernel.get_lambda()
        self.verbose = verbose
        self.device = device
        if device == "cuda":
            self.zero_arr = cp.zeros
            self.dtype = cp.float64
        else:
            self.zero_arr = np.zeros
            self.dtype = np.float64
        self.n_iter = 0
        self.n_updates = 0
        self.history_size = history_size

        self.losses = []
        self.preconditioner = preconditioner
        self.stored_mvecs = self.zero_arr((self.kernel.get_num_rffs(),
            dataset.get_n_classes(), self.history_size))
        self.stored_nconstants = self.zero_arr((self.history_size))
        self.stored_nconstants[:] = 1

        self.stored_bvecs = self.stored_mvecs.copy()
        self.stored_bconstants = self.stored_nconstants.copy()


    def fit_model(self, max_iter = 500, tol = 1e-4):
        """Finds an optimal set of weights using the information already
        provided to the class constructor.

        Args:
            max_iter (int): The maximum number of iterations for L_BFGS.
            tol (float): The threshold for convergence.

        Returns:
            wvec: A cupy or numpy array depending on device that contains the
                best set of weights found. A 1d array of length self.kernel.get_num_rffs().
        """
        wvec = self.zero_arr((self.kernel.get_num_rffs(), self.dataset.get_n_classes() ))
        self.n_iter, self.n_updates = 0, 0
        grad, loss = self.cost_fun_classification(wvec)
        self.losses = [loss]

        while self.n_iter < max_iter:
            grad, loss, wvec = self.update_params(grad, wvec, loss)
            self.losses.append(loss)
            if np.abs(np.abs(self.losses[-1] - self.losses[-2]) / self.losses[-2]) < tol:
                break
            self.n_iter += 1

        if self.device == "cuda":
            wvec = cp.asarray(wvec)
        return wvec, self.n_iter, self.losses




    def update_params(self, grad, wvec, loss):
        """Updates the weight vector and the approximate hessian maintained
        by the algorithm.

        Args:
            grad (ndarray): The previous gradient of the weights. Shape (num_rffs).
            wvec (ndarray): The current weight values.
            loss (float): The previous loss value.

        Returns:
            wvec (ndarray): The updated wvec.
            last_wvec (ndarray): Current wvec (which is now the last wvec).
        """
        altered_grad = self.inv_hess_vector_prod(grad)
        full_step_candidate_wvec = wvec - altered_grad

        # First consider using a step size (alpha) of 1. This will usually
        # be too much, at least initially. So interpolate the available
        # info using a quadratic and adjust step size accordingly.
        full_step_grad, full_step_loss = self.cost_fun_classification(
                full_step_candidate_wvec)
        alpha0_prime = -(grad * altered_grad).sum()
        step_size = -alpha0_prime / (2 * (full_step_loss - loss - alpha0_prime))
        new_wvec = wvec - step_size * altered_grad

        s_k = new_wvec - wvec

        new_grad, new_loss = self.cost_fun_classification(new_wvec)
        self.update_hess(new_grad - grad, s_k)
        if full_step_loss < loss:
            if new_loss < loss:
                return full_step_grad, full_step_loss, \
                        full_step_candidate_wvec
            # Returning the same loss that was input will
            # always automatically cause optimization to
            # terminate. Do this if neither proposed step
            # size is an improvement on the previous loss.
            return None, loss, wvec
        return new_grad, new_loss, new_wvec



    def update_hess(self, y_k, s_k):
        """Updates the hessian approximation.

        Args:
            y_k (ndarray): The shift in gradient on the current iteration.
            s_k (ndarray): The shift in the weight vector.
        """
        if self.n_updates >= self.history_size:
            return
        y_kh = self.inv_hess_vector_prod(y_k)
        s_kb = self.hess_vector_prod(s_k)

        h_update = s_k - y_kh
        b_update = y_k - s_kb
        criterion_lhs = np.abs(float(  (s_k * (y_k - s_kb)).sum()   ))
        criterion_rhs = 1e-8 * float( (s_k * s_k).sum()  ) * \
                float( (b_update * b_update).sum()  )

        if criterion_lhs >= criterion_rhs:
            self.stored_mvecs[:,:,self.n_updates] = h_update
            self.stored_nconstants[self.n_updates] = (h_update * y_k).sum()
            self.stored_bvecs[:,:,self.n_updates] = b_update
            self.stored_bconstants[self.n_updates] = (b_update * s_k).sum()
            self.n_updates += 1


    def inv_hess_vector_prod(self, ivec):
        """Takes the product of the inverse of the approximate
        Hessian with an input vector.

        Args:
            ivec (ndarray): A cupy or numpy array of shape (num_rffs, n_classes)
                with which to take the product.

        Returns:
            ovec (ndarray): Hk @ ivec.
        """
        if self.n_updates == 0:
            if self.preconditioner is None:
                return ivec
            return self.preconditioner.batch_matvec(ivec)

        ovec = (self.stored_mvecs[:,:,:self.n_updates] * ivec[:,:,None]).sum(axis=0).sum(axis=0) / \
                self.stored_nconstants[:self.n_updates]
        ovec = (ovec[None,None,:] * self.stored_mvecs[:,:,:self.n_updates]).sum(axis=2)
        if self.preconditioner is not None:
            ovec += self.preconditioner.batch_matvec(ivec)
        else:
            ovec += ivec
        return ovec


    def hess_vector_prod(self, ivec):
        """Takes the product of the approximate Hessian with
        an input vector.

        Args:
            ivec (ndarray): A cupy or numpy array of shape (num_rffs, n_classes)
                with which to take the product.

        Returns:
            ovec (ndarray): Hk @ ivec.
        """
        if self.n_updates == 0:
            if self.preconditioner is None:
                return ivec
            return self.preconditioner.rev_batch_matvec(ivec)

        ovec = (self.stored_bvecs[:,:,:self.n_updates] * ivec[:,:,None]).sum(axis=0).sum(axis=0) / \
                self.stored_bconstants[:self.n_updates]
        ovec = (ovec[None,None,:] * self.stored_bvecs[:,:,:self.n_updates]).sum(axis=2)
        if self.preconditioner is not None:
            ovec += self.preconditioner.rev_batch_matvec(ivec)
        else:
            ovec += ivec
        return ovec


    def cost_fun_classification(self, wvec):
        """The cost function for finding the weights for
        classification.

        Args:
            wvec (np.ndarray): A (num_rffs, 2) shape array. The first
                column is the current set of weights; the second
                is the proposed shift vector assuming step size 1.

        Returns:
            grad (ndarray): A cupy or numpy array containing the
                gradient, same shape as wvec.
            loss (float): A float indicating the current loss value.
        """
        ndatapoints = 0
        grad = self.zero_arr(wvec.shape)
        # We assume the first row is an intercept since fit_intercept will
        # always be set to true for classification.
        grad[1:,:] += self.lambda_**2 * wvec[1:,:]
        loss = 0.5 * self.lambda_**2 * (wvec**2)[1:,:].sum()

        for (xdata, ydata, ldata) in self.dataset.get_chunked_data():
            xd, yd = self.kernel.transform_x_y(xdata, ydata, ldata,
                    classification=True)
            ndatapoints += float(yd.shape[0])

            pred = xd @ wvec
            # Numerically stable softmax.
            pred -= pred.max(axis=1)[:,None]
            pred = 2.71828**pred
            pred /= pred.sum(axis=1)[:,None]
            if self.device == "cuda":
                logpred = cp.log(pred.clip(min=1e-16))
            else:
                logpred = np.log(pred.clip(min=1e-16))
            loss -= float(logpred[np.arange(pred.shape[0]), yd].sum())

            for k in range(wvec.shape[1]):
                if self.device == "cuda":
                    targets = (yd==k).astype(cp.float64)
                else:
                    targets = (yd==k).astype(np.float64)
                grad[:,k] += ((pred[:,k] - targets)[:,None] * xd).sum(axis=0)

        grad /= ndatapoints
        loss /= ndatapoints

        if self.verbose:
            print(f"Nfev {self.n_iter}, loss {loss}", flush=True)
        return grad, loss
