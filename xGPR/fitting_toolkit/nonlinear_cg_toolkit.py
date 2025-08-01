"""Contains the tools needed to get weights for the model
using the nonlinear CG algorithm with an optional
but strongly recommended preconditioner as H0."""
import numpy as np
try:
    import cupy as cp
except:
    pass



class nonlinear_CG_classification:
    """This class contains all the tools needed to fit a model
    whose hyperparameters have already been tuned using the
    nonlinear_CG algorithm. It is highly preferable to supply
    a preconditioner which acts as an H0 approximation, since
    otherwise this algorithm may take a long time to converge.
    It is intended for use with classification (just use
    straight CG for regression).

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
        losses (list): A list of loss values. Useful for comparing rate of
            convergence with other options.
        preconditioner: Either None or a valid preconditioner object.
        last_grad (ndarray): The last search direction
            taken. Used to generate the CG update.
    """

    def __init__(self, dataset, kernel, device, verbose,
            preconditioner = None, history_size = 5,
            bleed_in = 3):
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

        self.losses = []
        self.preconditioner = preconditioner
        self.last_grad = None
        self.last_search_direction = None


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
        self.n_iter = 0
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
        if self.preconditioner is not None:
            search_direction = self.preconditioner.batch_matvec(grad)
        else:
            search_direction = grad

        if self.last_grad is not None:
            polak_ribiere = (search_direction * (grad -
                self.last_grad)).sum()
            polak_ribiere /= (self.last_grad *
                    self.last_search_direction).sum()
            polak_ribiere = max(0., float(polak_ribiere))

            course_correction = polak_ribiere * self.last_search_direction
            self.last_grad = grad.copy()
            self.last_search_direction = search_direction.copy()
            search_direction += course_correction
        else:
            self.last_grad = grad.copy()
            self.last_search_direction = search_direction.copy()

        search_direction = -search_direction
        full_step_candidate_wvec = wvec + search_direction

        # First consider using a step size (alpha) of 1. This will usually
        # be too much, at least initially. So interpolate the available
        # info using a quadratic and adjust step size accordingly.
        full_step_grad, full_step_loss = self.cost_fun_classification(
                full_step_candidate_wvec)
        alpha0_prime = (grad * search_direction).sum()
        quad_step_size = -alpha0_prime / (2 * (full_step_loss - loss - alpha0_prime))
        quad_wvec = wvec + quad_step_size * search_direction

        quad_grad, quad_loss = self.cost_fun_classification(quad_wvec)

        if quad_loss < full_step_loss and quad_loss < loss:
            return quad_grad, quad_loss, quad_wvec

        # If quadratic interpolation was unsuccessful,
        # interpolate using a cubic. This can be further
        # extended to do a backtracking Armijo line
        # search...
        a_term = quad_loss - loss - (alpha0_prime * quad_step_size) - \
                quad_step_size**2 * \
                (full_step_loss - loss - alpha0_prime)
        b_term = -(quad_loss - loss - (alpha0_prime * quad_step_size)) + \
                quad_step_size**3 * \
                (full_step_loss - loss - alpha0_prime)
        divisor = 1 / (quad_step_size**2 *
            (quad_step_size - 1))
        b_term /= divisor
        a_term /= divisor
        cubic_step_size = (-b_term + np.sqrt(b_term**2 -
            3 * a_term * alpha0_prime)) / (3 * a_term)
        cubic_wvec = wvec - cubic_step_size * search_direction
        cubic_grad, cubic_loss = self.cost_fun_classification(cubic_wvec)

        if cubic_loss < full_step_loss and cubic_loss < loss:
            return cubic_grad, cubic_loss, cubic_wvec

        if full_step_loss < loss:
            return full_step_grad, full_step_loss, \
                    full_step_candidate_wvec
        # Returning the same loss that was input will
        # always automatically cause optimization to
        # terminate. Do this if neither proposed step
        # size is an improvement on the previous loss.
        return None, loss, wvec


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
