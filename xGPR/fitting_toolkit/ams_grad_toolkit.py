"""Contains the tools needed to get weights for the model
using the AMS-Grad optimization algorithm, which is
currently used only for testing."""
import numpy as np
#cimport numpy as np
try:
    import cupy as cp
except:
    pass

class amsModelFit:
    """This class contains all the tools needed to fit a model
    whose hyperparameters have already been tuned using AMSGrad.
    Currently for testing purposes only, this has not been shown
    to work well on tough problems.

    Attributes:
        lambda_ (float): The noise hyperparameter shared across all kernels.
        verbose (bool): If True, print regular updates.
        device_ (str): One of 'cpu', 'gpu'. Indicates where calculations
            will be performed.
        n_epoch (int): The number of epochs.
        n_iter (int): The number of datapoints traversed in current epoch.
    """

    def __init__(self, lambda_, device, verbose):
        """Class constructor.

        Args:
            lambda_ (float): The noise hyperparameter shared across all kernels.
            device (str): One of 'cpu', 'gpu'. Indicates where calculations
                will be performed.
            verbose (bool): If True, print regular updates.
        """
        self.lambda_ = lambda_
        self.verbose = verbose
        self.device = device
        self.n_iter = 0
        self.n_epoch = 0
        self.n_iter = 0

    def fit_model(self, dataset, kernel,
            tol = 1e-6, max_epochs = 40):
        """Finds an optimal set of weights using the information already
        provided to the class constructor.

        Args:
            dataset: A Dataset object.
            kernel: A kernel object that can generate random features for
                the Dataset.
            tol (float): The threshold for convergence.
            max_epochs (int): The number of epochs.

        Returns:
            wvec: A cupy or numpy array depending on device that contains the
                best set of weights found. A 1d array of length self.kernel.get_num_rffs().
        """
        #Key sgd hyperparameters.
        beta1 = 0.9
        beta2 = 0.999
        step_size = 1e-12
        h_lr = 1e-9
        losses = []

        wvec = np.zeros((kernel.get_num_rffs()))
        v_t = np.zeros((kernel.get_num_rffs(),2))
        sqrt = np.sqrt
        if self.device == "gpu":
            wvec = cp.asarray(wvec)
            v_t = cp.asarray(v_t)
            sqrt = cp.sqrt
        m_t, grad, update = wvec.copy(), wvec.copy(), v_t.copy()

        for self.n_epoch in range(max_epochs):
            for xdata, ydata in dataset.get_chunked_data():
                if not dataset.pretransformed:
                    xdata = kernel.transform_x(xdata)
                grad[:] = xdata.T @ (xdata @ wvec)
                grad[:] += self.lambda_**2 * wvec
                grad[:] -= xdata.T @ ydata

                m_t = beta1 * m_t + (1 - beta1) * grad
                v_t[:,0] = beta2 * v_t[:,1] + (1 - beta2) * grad**2
                v_t[:,0] = v_t.max(axis=1)

                update[:,0] = m_t / sqrt(v_t[:,0] + 1e-14)
                wvec -= step_size * update[:,0]
                step_size += h_lr * (grad * update[:,1]).sum()
                step_size = min(max(step_size, 1e-12), 1e-3)
                v_t[:,1] = v_t[:,0]
                update[:,1] = update[:,0]

            full_grad = self.update_full_gradient(dataset, kernel, wvec)
            losses.append( float(full_grad.T @ full_grad) )

            if losses[-1] < tol:
                break

            if self.verbose and self.n_epoch % 1 == 0:
                print(f"Epoch {self.n_epoch} complete; loss {losses[-1]}")
        return wvec, losses


    def update_full_gradient(self, dataset, kernel, wvec):
        """Gets the full, unapproximated loss. Used for testing.
        (Currently AMSGrad is ONLY used for testing, so this
        is always used."""
        if self.device == "gpu":
            full_grad = cp.zeros((kernel.get_num_rffs()))
        else:
            full_grad = np.zeros((kernel.get_num_rffs()))
        z_trans_y = full_grad.copy()
        for xdata, ydata in dataset.get_chunked_data():
            if not dataset.pretransformed:
                xdata = kernel.transform_x(xdata)
            full_grad += xdata.T @ (xdata @ wvec)
            z_trans_y += xdata.T @ ydata
        zty_norm = np.sqrt(float(full_grad.T @ full_grad))
        full_grad += self.lambda_**2 * wvec - z_trans_y
        return full_grad / zty_norm


    def get_niter(self):
        """Returns the number of function evaluations performed."""
        return self.n_epoch
