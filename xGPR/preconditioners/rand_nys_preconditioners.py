"""Implements a randomized nystrom preconditioner with
some methods that are useful for calculating NMLL
during tuning."""
try:
    import cupy as cp
except:
    pass
import numpy as np

from .rand_nys_constructors import initialize_srht, initialize_srht_multipass



class RandNysPreconditioner():
    """Implements a preconditioner based on the randomized
    Nystrom approximation to the inverse of Z^T Z + lambda.
    """
    def __init__(self, kernel, dataset, max_rank, verbose,
                random_state = 123, method = "srht",
                class_means=None, class_weights=None):
        """Class constructor.

        Args:
            kernel: A valid kernel object that can generate random features.
            dataset: A valid online or offline dataset object that can retrieve
                chunked data.
            max_rank (int): The rank of the preconditioner.
            verbose (bool): If True, print regular updates.
            random_state (int): The random number seed for the random number
                generator.
            method (str): One of "srht", "srht_2". Determines the method of
                preconditioner construction.
            class_means: Either None or a (nclasses, num_rffs)
                array storing the mean of the features for each
                class.
            class_weights: Either None or an (nclasses) array storing
                the class_weight for each class.
        """
        if method not in ["srht_2", "srht_3", "srht"]:
            raise RuntimeError("Unknown method supplied for tuning preconditioner "
                    "construction.")

        if method.startswith("srht_"):
            n_passes = int(method.split("_")[1])
            self.u_mat, self.eig, self.z_trans_y, self.y_trans_y = \
                            initialize_srht_multipass(dataset,
                                max_rank, kernel, random_state, verbose,
                                n_passes, class_means=class_means,
                                class_weights=class_weights)
        else:
            self.u_mat, self.eig, self.z_trans_y, self.y_trans_y = \
                            initialize_srht(dataset, max_rank,
                                kernel, random_state, verbose,
                                class_means=class_means,
                                class_weights=class_weights)

        lambda_ = kernel.get_lambda()
        min_eig = self.eig.min()
        self.eig += lambda_**2

        self.inv_eig = self.eig.copy()
        mask = self.inv_eig > 1e-14
        self.inv_eig[mask] = 1 / self.inv_eig[mask]
        self.inv_eig[mask==False] = 0.0

        self.achieved_ratio = min_eig / lambda_**2
        self.prefactor = float(min_eig + lambda_**2)

        self.device = kernel.device


    def batch_matvec(self, xvec):
        """Returns a matvec of the preconditioner with a set of input
        vectors xvec."""
        xprod = self.u_mat.T @ xvec
        xprod1 = self.u_mat @ (self.inv_eig[:,None] * self.prefactor * xprod)
        xprod2 = xvec - (self.u_mat @ xprod)
        return xprod2 + xprod1

    def rev_batch_matvec(self, xvec):
        """Returns a matvec of the non-inverted preconditioner with a set of input
        vectors xvec."""
        xprod = self.u_mat.T @ xvec
        xprod1 = self.u_mat @ (self.eig[:,None] * xprod) / self.prefactor
        xprod2 = xvec - (self.u_mat @ xprod)
        return xprod2 + xprod1


    def _matvec(self, xvec):
        """Implements the matvec for a single input vector."""
        xprod = self.u_mat.T @ xvec
        xprod1 = self.u_mat @ (self.inv_eig * self.prefactor * xprod)
        xprod2 = xvec - (self.u_mat @ xprod)
        return xprod2 + xprod1


    def get_rank(self):
        """Returns the preconditioner rank."""
        return self.inv_eig.shape[0]


    def get_logdet(self):
        """Returns the log determinant of the preconditioner. Useful
        for some hyperparameter tuning schemes."""
        logdet = 1 + (self.eig - self.prefactor) / self.prefactor
        if self.device == "cpu":
            return float(np.log(logdet.clip(min=1e-12)).sum())
        return float(cp.log(logdet.clip(min=1e-12)).sum())


    def matvec_for_sampling(self, xvec):
        """Returns self.u_mat @ eigvals[:,None] * x, where x is of shape
        (self.eig.shape[0], nsamples) and eigvals is the square root of
        the eigenvalues of the first term of the preconditioner."""
        eigvals = self.eig.copy()
        eigvals[eigvals < 0] = 0
        if self.device == "cpu":
            eigvals[:] = np.sqrt(eigvals)
        else:
            eigvals[:] = cp.sqrt(eigvals)
        prefactor = np.sqrt(1 / self.prefactor)
        xprod = self.u_mat.T @ xvec
        xprod1 = self.u_mat @ (eigvals[:,None] * prefactor * xprod)
        xprod2 = xvec - (self.u_mat @ xprod)
        return xprod1 + xprod2

    def get_zty(self):
        """Returns the product Z^T y, which is assembled during
        preconditioner construction for regression models. If
        this preconditioner was created for a classification
        model, z_trans_y will be None."""
        return self.z_trans_y


    def get_yty(self):
        """Returns the product y^T y, which is assembled during
        preconditioner construction."""
        return float(self.y_trans_y)


    @property
    def device(self):
        """Getter for the device property, which determines
        whether calculations are on CPU or GPU."""
        return self.device_


    @device.setter
    def device(self, value):
        """Setter for device, which determines whether calculations
        are on CPU or GPU.

        Args:
            value (str): Must be one of 'cpu', 'cuda'.

        Raises:
            RuntimeError: A RuntimeError is raised if an unrecognized
                device is passed.
        """
        if value == "cpu":
            if not isinstance(self.u_mat, np.ndarray):
                self.u_mat = cp.asnumpy(self.u_mat)
                self.inv_eig = cp.asnumpy(self.inv_eig)
                # Note that for classification models z_trans_y is
                # not stored / not needed and may therefore be
                # None.
                if self.z_trans_y is not None:
                    self.z_trans_y = cp.asnumpy(self.z_trans_y)

        elif value == "cuda":
            self.u_mat = cp.asarray(self.u_mat)
            self.inv_eig = cp.asarray(self.inv_eig)
            # Note that for classification models z_trans_y is
            # not stored / not needed and may therefore be
            # None.
            if self.z_trans_y is not None:
                self.z_trans_y = cp.asarray(self.z_trans_y)

        else:
            raise RuntimeError("Unrecognized device supplied. Must be one "
                    "of 'cpu', 'cuda'.")

        self.device_ = value
