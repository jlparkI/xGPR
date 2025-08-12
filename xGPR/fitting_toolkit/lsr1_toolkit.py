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
        n_iter (int): The number of function evaluations performed.
        n_updates (int): The number of Hessian updates to date. Not necessarily
            the same as n_iter, since Hessian updates can be skipped.
        history_size (int): The number of previous gradient updates to store.
        current_replacement_token (int): The element of the approximate Hessian
            that should be replaced on next update.
        losses (list): A list of loss values. Useful for comparing rate of
            convergence with other options.
        preconditioner: Either None or a valid preconditioner object.
        stored_mvecs (ndarray): A cupy or numpy array containing the
            sk - Hk yk terms; shape (num_rffs, history_size).
        stored_nconstants (ndarray): The denominator of the Hessian
            update; shape (history_size).
    """

    def __init__(self, dataset, kernel, device, verbose,
            preconditioner = None, history_size = 5):
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
        else:
            self.zero_arr = np.zeros
        self.n_iter = 0
        self.n_updates = 0
        self.history_size = history_size
        self.current_replacement_token = 0

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
        search_direction = -self.inv_hess_vector_prod(grad)
        alpha0_prime = (grad * search_direction).sum()

        # First consider using a step size (alpha) of 1.
        full_step_wvec = wvec + search_direction
        full_step_grad, full_step_loss = self.cost_fun_classification(
                full_step_wvec)
        # If this initial guess satisfies the Armijo conditions, halt.
        if full_step_loss < (loss + 1e-4 * alpha0_prime):
            return full_step_grad, full_step_loss, full_step_wvec

        # Otherwise interpolate using a quadratic, which often works well,
        # especially on initial steps.
        quad_step_size = -alpha0_prime / (2 * (full_step_loss - loss - alpha0_prime))
        quad_wvec = wvec + quad_step_size * search_direction
        quad_grad, quad_loss = self.cost_fun_classification(quad_wvec)
        # If this guess satisfies the Armijo conditions, halt.
        if quad_loss < (loss + quad_step_size * 1e-4 * alpha0_prime):
            return quad_grad, quad_loss, quad_wvec

        # If quadratic interpolation was unsuccessful,
        # do a backtracking line search. This is expensive so we try
        # the much cheaper easy guesses first.
        losses = [loss, full_step_loss, quad_loss]
        grads = [grad, full_step_grad, quad_grad]
        wvecs = [wvec, full_step_wvec, quad_wvec]

        alpha_max, rfactor = 1, 0.5
        if quad_loss < full_step_loss:
            alpha_max = quad_step_size

        # Run up to 10 iterations. At that point if we haven't found anything
        # it's best to cut our losses.
        for _ in range(10):
            alpha = rfactor * alpha_max
            candidate_wvec = wvec + alpha * search_direction
            candidate_grad, candidate_loss = self.cost_fun_classification(
                candidate_wvec)
            # If the current step size worked, stop here.
            if candidate_loss < (loss + alpha * 1e-4 * alpha0_prime):
                return candidate_grad, candidate_loss, candidate_wvec

            # Otherwise, keep track of this update and continue.
            losses.append(candidate_loss)
            grads.append(candidate_grad)
            wvecs.append(candidate_wvec)

            rfactor *= 0.5


        # If we're still here, the backtracking line search never met Armijo
        # conditions. Return the best candidate we found.
        best_idx = np.argmin(losses)
        return grads[best_idx], losses[best_idx], wvecs[best_idx]




    def update_hess(self, y_k, s_k):
        """Updates the hessian approximation.

        Args:
            y_k (ndarray): The shift in gradient on the current iteration.
            s_k (ndarray): The shift in the weight vector.

        Returns:
            success (int): 0 if failure, 1 if success.
        """
        y_kh = self.inv_hess_vector_prod(y_k)
        s_kb = self.hess_vector_prod(s_k)

        h_update = s_k - y_kh
        b_update = y_k - s_kb
        criterion_lhs = np.abs(float(  (s_k * (y_k - s_kb)).sum()   ))
        criterion_rhs = 1e-8 * np.sqrt(float( (s_k * s_k).sum()  )) * \
                np.sqrt(float( (b_update * b_update).sum()  ))

        if criterion_lhs >= criterion_rhs:
            self.stored_mvecs[:,:,self.current_replacement_token] = h_update
            self.stored_nconstants[self.current_replacement_token] = (h_update * y_k).sum()
            self.stored_bvecs[:,:,self.current_replacement_token] = b_update
            self.stored_bconstants[self.current_replacement_token] = (b_update * s_k).sum()
            if self.n_updates < self.history_size:
                self.n_updates += 1
            self.current_replacement_token += 1
            if self.current_replacement_token >= self.history_size:
                self.current_replacement_token = 0


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

        # The first version essentially has a separate hessian for every class.
        #ovec = (self.stored_mvecs[:,:,:self.n_updates] * ivec[:,:,None]).sum(axis=0) / \
        #        self.stored_nconstants[None, :self.n_updates]
        #ovec = (ovec[None,:,:] * self.stored_mvecs[:,:,:self.n_updates]).sum(axis=2)
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

        #ovec = (self.stored_bvecs[:,:,:self.n_updates] * ivec[:,:,None]).sum(axis=0) / \
        #        self.stored_bconstants[None,:self.n_updates]
        #ovec = (ovec[None,:,:] * self.stored_bvecs[:,:,:self.n_updates]).sum(axis=2)
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
        grad = self.zero_arr(wvec.shape)
        # We assume the first row is an intercept since fit_intercept will
        # always be set to true for classification.
        grad[1:,:] += self.lambda_**2 * wvec[1:,:]
        loss = 0.5 * self.lambda_**2 * (wvec**2)[1:,:].sum()

        for (xdata, ydata, ldata) in self.dataset.get_chunked_data():
            xd, yd = self.kernel.transform_x_y(xdata, ydata, ldata,
                    classification=True)

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

        if self.verbose:
            print(f"Niter {self.n_iter}, loss {loss}", flush=True)
        return grad, loss
