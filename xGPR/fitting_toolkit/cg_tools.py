"""A conjugate gradients implementation, set up to work with
a batched right-hand side. Unlike cupy's / scipy's it offers us
the ability to easily have multiple outputs if needed."""
import numpy as np
try:
    import cupy as cp
except:
    pass






class GPU_ConjugateGrad:
    """Performs conjugate gradients to find b in Ab = y.
    Used both for fitting and for NMLL calculations via
    SLQ.
    """

    def __init__(self, class_means=None, class_weights=None):
        self.class_means = class_means
        self.class_weights = class_weights


    def _matvec(self, dataset, kernel, vec, matvec):
        """Performs a matvec operation (Z^T Z + lambda**2) vec,
        where Z is the random features for the raw data
        from dataset.

        Args:
            dataset: An OnlineDataset or OfflineDataset object
                with the raw data.
            kernel: A valid kernel object.
            vec (array): The target array.
            matvec (array): The product (Z^T Z + lambda**2) vec.
                This array is modified in-place.
        """
        matvec[:] = 0
        # If there are no class means, we are doing regression.
        if self.class_means is None:
            for x, lengths in dataset.get_chunked_x_data():
                Z = kernel.transform_x(x, lengths)
                matvec += Z.T @ (Z @ vec)
        # Otherwise, classification. The features need to be modified
        # using the class means.
        else:
            for x, ydata, lengths in dataset.get_chunked_data():
                Z, _ = kernel.transform_x_y(x, ydata, lengths,
                        self.class_means, self.class_weights,
                        classification=True)
                matvec += Z.T @ (Z @ vec)
        matvec += kernel.get_lambda()**2 * vec



    def fit(self, dataset, kernel, preconditioner,
            resid, maxiter = 200, tol = 1e-4,
            verbose = True,
            nmll_settings = False):
        """Performs conjugate gradients to evaluate (Z^T Z + lambda)^-1 Z^T y,
        where Z is the random features generated for the dataset and lambda
        is the shared noise hyperparameter.

        Args:
            dataset: An OnlineDataset or OfflineDataset with the raw data.
            kernel: A valid kernel object.
            preconditioner: Either None or a valid preconditioner object.
            resid (cp.ndarray): The initial residuals, using starting
                weights of 0 (this CG function always assumes that
                starting weights are all 0).
            maxiter (int): The maximum number of iterations.
            tol (double): The threshold for convergence.
            verbose (bint): If True, print regular updates.
            nmll_settings (bint): If True, save and
                return the alpha and beta values (used for NMLL calcs).
                If False, return the number of iterations and a list of
                losses (for diagnostic purposes).

        Returns:
            xk (cp.ndarray): The result of (Z^T Z + lambda)^-1 Z^T y.
            converged (bint): If False, we did not converge, just ran
                into maxiter.
            niter: The number of iterations.
        """
        converged = False

        target = resid[:,0,:].copy()
        init_norms = cp.linalg.norm(target, axis=0)

        z_k = cp.zeros((resid.shape[0], 2, resid.shape[2]), dtype=cp.float64)
        p_k = cp.zeros((resid.shape[0], 2, resid.shape[2]), dtype=cp.float64)

        alpha = cp.zeros((resid.shape[2]), dtype=cp.float64)
        beta = cp.zeros((resid.shape[2]), dtype=cp.float64)
        alphas, betas, losses = [], [], []

        x_k = cp.zeros((resid.shape[0], resid.shape[2]), dtype=cp.float64)
        w = x_k.copy()

        if preconditioner is None:
            z_k[:,0,:] = resid[:,0,:]
        else:
            z_k[:,0,:] = preconditioner.batch_matvec(resid[:,0,:])
        p_k[:,0,:] = z_k[:,0,:]

        next_col, current_col = 1, 0
        for niter in range(maxiter):
            self._matvec(dataset, kernel, p_k[:,current_col,:], w)
            alpha[:] = (resid[:,current_col,:] *
                    z_k[:,current_col,:]).sum(axis=0) / \
                    (p_k[:,current_col,:] * w).sum(axis=0)
            x_k += alpha[None,:] * p_k[:,current_col,:]
            resid[:,next_col,:] = resid[:,current_col,:] - \
                        alpha[None,:] * w
            err = cp.linalg.norm(resid[:,current_col,:], axis=0) / init_norms

            if preconditioner is None:
                z_k[:,next_col,:] = resid[:,next_col,:]
            else:
                z_k[:,next_col,:] = preconditioner.batch_matvec(resid[:,next_col,:])

            beta[:] = (resid[:,next_col,:] * z_k[:,next_col,:]).sum(axis=0) / \
                    (resid[:,current_col,:] * z_k[:,current_col,:]).sum(axis=0)
            p_k[:,next_col,:] = z_k[:,next_col,:] + beta[None,:] * \
                    p_k[:,current_col,:]

            if nmll_settings:
                alphas.append(alpha.copy())
                betas.append(beta.copy())
            else:
                losses.append(float(err[0]))

            next_col = abs(next_col - 1)
            current_col = abs(current_col - 1)

            if niter % 5 == 0 and verbose:
                print(f"{niter} iterations complete.")
            if err.max() < tol:
                converged = True
                break

        if nmll_settings:
            if len(alphas) > 1:
                alphas, betas = cp.stack(alphas), cp.stack(betas)
            else:
                alphas = alphas[0].reshape(1, alphas[0].shape[0])
                betas = betas[0].reshape(1, betas[0].shape[0])
            alphas, betas = cp.asnumpy(alphas[:,1:]), cp.asnumpy(betas[:,1:])
            return x_k, alphas, betas
        
        # This is true if and only if we are doing classification.
        if x_k.shape[1] > 1:
            return x_k, converged, niter + 1, losses

        return x_k[:,0], converged, niter + 1, losses





class CPU_ConjugateGrad:
    """Performs conjugate gradients to find b in Ab = y.
    Used both for fitting and for NMLL calculations via
    SLQ.
    """

    def __init__(self, class_means=None, class_weights=None):
        self.class_means = class_means
        self.class_weights = class_weights


    def _matvec(self, dataset, kernel, vec, matvec):
        """Performs a matvec operation (Z^T Z + lambda**2) vec,
        where Z is the random features for the raw data
        from dataset.

        Args:
            dataset: An OnlineDataset or OfflineDataset object
                with the raw data.
            kernel: A valid kernel object.
            vec (array): The target array.
            matvec (array): The product (Z^T Z + lambda**2) vec.
                This array is modified in-place.
        """
        matvec[:] = 0
        # If there are no class means, we are doing regression.
        if self.class_means is None:
            for x, lengths in dataset.get_chunked_x_data():
                Z = kernel.transform_x(x, lengths)
                matvec += Z.T @ (Z @ vec)
        # Otherwise, classification. The features need to be modified
        # using the class means.
        else:
            for x, ydata, lengths in dataset.get_chunked_data():
                Z, _ = kernel.transform_x_y(x, ydata, lengths,
                        self.class_means, self.class_weights,
                        classification=True)
                matvec += Z.T @ (Z @ vec)
        matvec += kernel.get_lambda()**2 * vec


    def fit(self, dataset, kernel, preconditioner,
            resid, maxiter = 200, tol = 1e-4,
            verbose = True,
            nmll_settings = False):
        """Performs conjugate gradients to evaluate (Z^T Z + lambda)^-1 Z^T y,
        where Z is the random features generated for the dataset and lambda
        is the shared noise hyperparameter.

        Args:
            dataset: An OnlineDataset or OfflineDataset with the raw data.
            kernel: A valid kernel object.
            preconditioner: Either None or a valid preconditioner object.
            resid (np.ndarray): The initial residuals, using starting
                weights of 0 (this CG function always assumes that
                starting weights are all 0).
            maxiter (int): The maximum number of iterations.
            tol (double): The threshold for convergence.
            verbose (bint): If True, print regular updates.
            nmll_settings (bint): If True, save and
                return the alpha and beta values (used for NMLL calcs).
                If False, return the number of iterations and a list of
                losses (for diagnostic purposes).

        Returns:
            xk (np.ndarray): The result of (Z^T Z + lambda)^-1 Z^T y.
            converged (bint): If False, we did not converge, just ran
                into maxiter.
            niter: The number of iterations.
        """
        converged = False

        target = resid[:,0,:].copy()
        init_norms = np.linalg.norm(target, axis=0)

        z_k = np.zeros((resid.shape[0], 2, resid.shape[2]), dtype=np.float64)
        p_k = np.zeros((resid.shape[0], 2, resid.shape[2]), dtype=np.float64)

        alpha = np.zeros((resid.shape[2]), dtype=np.float64)
        beta = np.zeros((resid.shape[2]), dtype=np.float64)
        alphas, betas, losses = [], [], []

        x_k = np.zeros((resid.shape[0], resid.shape[2]), dtype=np.float64)
        w = x_k.copy()

        if preconditioner is None:
            z_k[:,0,:] = resid[:,0,:]
        else:
            z_k[:,0,:] = preconditioner.batch_matvec(resid[:,0,:])

        p_k[:,0,:] = z_k[:,0,:]

        next_col, current_col = 1, 0
        for niter in range(maxiter):
            self._matvec(dataset, kernel, p_k[:,current_col,:], w)
            alpha[:] = (resid[:,current_col,:] *
                    z_k[:,current_col,:]).sum(axis=0) / \
                    (p_k[:,current_col,:] * w).sum(axis=0)
            x_k += alpha[None,:] * p_k[:,current_col,:]
            resid[:,next_col,:] = resid[:,current_col,:] - \
                        alpha[None,:] * w
            resid[:,next_col,:] = resid[:,current_col,:] - \
                    alpha[None,:] * w
            err = np.linalg.norm(resid[:,current_col,:], axis=0) / init_norms

            if preconditioner is None:
                z_k[:,next_col,:] = resid[:,next_col,:]
            else:
                z_k[:,next_col,:] = preconditioner.batch_matvec(resid[:,next_col,:])
            beta[:] = (resid[:,next_col,:] * z_k[:,next_col,:]).sum(axis=0) / \
                    (resid[:,current_col,:] * z_k[:,current_col,:]).sum(axis=0)
            p_k[:,next_col,:] = z_k[:,next_col,:] + beta[None,:] * \
                    p_k[:,current_col,:]

            if nmll_settings:
                alphas.append(alpha.copy())
                betas.append(beta.copy())
            else:
                losses.append(float(err[0]))

            next_col = abs(next_col - 1)
            current_col = abs(current_col - 1)

            if niter % 5 == 0 and verbose:
                print(f"{niter} iterations complete.")
            if err.max() < tol:
                converged = True
                break

        if nmll_settings:
            if len(alphas) > 1:
                alphas, betas = np.stack(alphas), np.stack(betas)
            else:
                alphas = alphas[0].reshape(1, alphas[0].shape[0])
                betas = betas[0].reshape(1, betas[0].shape[0])
            return x_k, alphas[:,1:], betas[:,1:]

        # This is true if and only if we are doing classification.
        if x_k.shape[1] > 1:
            return x_k, converged, niter + 1, losses
        return x_k[:,0], converged, niter + 1, losses
