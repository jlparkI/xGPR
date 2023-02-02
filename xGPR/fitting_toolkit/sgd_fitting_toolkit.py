"""Contains the tools needed to get weights for the model
using the L-BFGS optimization algorithm."""
import numpy as np
try:
    import cupy as cp
except:
    pass

class sgdModelFit:
    """This class contains all the tools needed to fit a model
    whose hyperparameters have already been tuned using implicit
    SGD. This will be slower than preconditioned CG if the
    preconditioner is good (low ratio), but can outperform
    preconditioned CG with a high-ratio preconditioner.

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
        self.mbatch_size = 250
        if self.device == "cpu":
            self.empty = np.empty
            self.zeros = np.zeros
        else:
            self.empty = cp.empty
            self.zeros = cp.zeros


    def fit_model(self, dataset, kernel, tol = 1e-6, max_epochs = 40,
            preconditioner = None, manual_lr = None):
        """Finds an optimal set of weights using the information already
        provided to the class constructor.

        Args:
            dataset: An OnlineDataset or OfflineDatset containing all the
                training data.
            kernel: A kernel object that can generate random features for
                the Dataset.
            tol (float): The threshold for convergence.
            max_epochs (int): The number of epochs. Used to set the learning
                rate schedule.
            random_state (int): A seed for the random number generator.
            manual_lr (float): Either None or a float. If not None, this
                is a user-specified initial learning rate. If None, find
                a good initial learning rate using autotuning.
            mbatch_lr_check (int): The number of minibatches after which
                to check that the loss is not diverging (and reset the
                learning rate if it is).

        Returns:
            wvec: A cupy or numpy array depending on device that contains the
                best set of weights found. A 1d array of length self.kernel.get_num_rffs().
        """
        losses = []

        #Key sgd hyperparameters.
        likely_lr = [3**i * 1e-9 for i in range(25)]
        if self.device == "cpu":
            likely_lr = np.asarray(likely_lr)
        else:
            likely_lr = cp.asarray(likely_lr)


        full_grad, wvec, z_trans_y, zty_norm = self.initialize(dataset, kernel)

        current_grad, last_wvec = full_grad.copy(), wvec.copy()
        sg_last_grad, sg_current_grad = self.zeros((kernel.get_num_rffs())), \
                self.zeros((kernel.get_num_rffs()))
        if manual_lr is not None:
            step_size = manual_lr
        else:
            step_size = self.autotune(dataset, kernel, full_grad, wvec,
                    last_wvec, preconditioner, z_trans_y, likely_lr)


        for self.n_epoch in range(max_epochs):
            end_epoch = False
            while not end_epoch:
                xbatch, _, end_epoch = dataset.get_next_minibatch(self.mbatch_size)
                if not dataset.pretransformed:
                    xbatch = kernel.transform_x(xbatch)
                sg_current_grad[:] = xbatch.T @ (xbatch @ wvec) + self.lambda_**2 * wvec
                sg_last_grad[:] = xbatch.T @ (xbatch @ last_wvec) + self.lambda_**2 * last_wvec

                current_grad[:] = full_grad + (sg_current_grad - sg_last_grad)
                if preconditioner is not None:
                    wvec -= step_size * preconditioner.batch_matvec(current_grad[:,None])[:,0]
                else:
                    wvec -= step_size * current_grad / dataset.get_ndatapoints()


            dataset.reset_index()
            self.update_full_gradient(dataset, kernel, full_grad,
                            wvec, z_trans_y)
            last_wvec[:] = wvec
            current_grad[:] = full_grad
            loss = full_grad / zty_norm
            loss = np.sqrt(float( loss.T @ loss ) )
            losses.append(loss)
            if len(losses) > 2:
                if losses[-1] - losses[-2] > 0:
                    print("Reducing learning rate by 50%")
                    step_size *= 0.5

            if losses[-1] < tol:
                break

            if self.verbose and self.n_epoch % 1 == 0:
                print(f"Epoch {self.n_epoch} complete; loss {losses[-1]}")
        return wvec.copy(), losses



    def initialize(self, dataset, kernel):
        full_grad = self.zeros((kernel.get_num_rffs()))
        wvec = self.zeros((kernel.get_num_rffs()))
        z_trans_y = self.zeros((kernel.get_num_rffs()))
        zty_norm = 0.0

        for xdata, ydata in dataset.get_chunked_data():
            if not dataset.pretransformed:
                xdata = kernel.transform_x(xdata)
            z_trans_y += xdata.T @ ydata

        zty_norm = np.sqrt(float(z_trans_y.T @ z_trans_y))
        full_grad[:] = -z_trans_y
        return full_grad, wvec, z_trans_y, zty_norm


    def update_full_gradient(self, dataset, kernel, full_grad, wvec,
            z_trans_y):
        full_grad[:] = -z_trans_y + self.lambda_**2 * wvec
        for xdata in dataset.get_chunked_x_data():
            if not dataset.pretransformed:
                xdata = kernel.transform_x(xdata)
            full_grad += xdata.T @ (xdata @ wvec)


    def autotune(self, dataset, kernel, full_grad, wvec,
            last_wvec, precond, z_trans_y, likely_lr):
        """Uses a simple heuristic to tune the learning rate over the first 10
        minibatches in an arbitrarily designated epoch. Maybe not the best
        possible way to do this, but seems to work..."""
        wvec_batch = self.empty((kernel.get_num_rffs(), likely_lr.shape[0]))
        last_wvec_batch = self.empty((kernel.get_num_rffs(), likely_lr.shape[0]))
        gradient_batch = self.empty((kernel.get_num_rffs(), likely_lr.shape[0]))
        losses = self.zeros((kernel.get_num_rffs(), likely_lr.shape[0]))
        wvec_batch[:] = wvec[:,None]
        last_wvec_batch[:] = last_wvec[:,None]

        #Recall that we check under "fit" that the dataset has at least
        #10 minibatches...Using more might lead to more accurate tuning
        #but would make tuning more expensive...it's a tradeoff.
        end_epoch = False
        while not end_epoch:
            gradient_batch[:] = full_grad[:,None]
            xbatch, _, end_epoch = dataset.get_next_minibatch(self.mbatch_size)
            if not dataset.pretransformed:
                xbatch = kernel.transform_x(xbatch)
            gradient_batch += xbatch.T @ (xbatch @ wvec_batch)
            gradient_batch -= xbatch.T @ (xbatch @ last_wvec_batch)
            gradient_batch += kernel.get_lambda()**2 * (wvec_batch - last_wvec_batch)

            if precond is not None:
                wvec_batch -= likely_lr * precond.batch_matvec(gradient_batch)
            else:
                wvec_batch -= likely_lr[None,:] * gradient_batch / dataset.get_ndatapoints()

        dataset.reset_index()

        losses[:] = -z_trans_y[:,None] + self.lambda_**2 * wvec_batch
        for xbatch in dataset.get_chunked_x_data():
            if not dataset.pretransformed:
                xbatch = kernel.transform_x(xbatch)
            losses += xbatch.T @ (xbatch @ wvec_batch)


        losses[np.isnan(losses)] = np.inf
        losses = (losses**2).sum(axis=0)
        best_idx = int(losses.argmin())
        return likely_lr[best_idx]



    def get_niter(self):
        """Returns the number of function evaluations performed."""
        return 2 * self.n_epoch + 1
