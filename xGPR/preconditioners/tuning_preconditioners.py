"""Implements a randomized nystrom preconditioner with
some methods that are useful for calculating NMLL
during tuning."""
try:
    import cupy as cp
except:
    pass
import numpy as np

from .rand_nys_constructors import initialize_srht, initialize_srht_multipass



class RandNysTuningPreconditioner():
    """Implements a preconditioner based on the randomized
    Nystrom approximation to the inverse of Z^T Z + lambda
    with some methods that are useful when tuning.
    """
    def __init__(self, kernel, dataset, max_rank, verbose,
                random_state = 123, method = "srht"):
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
        """
        self.device = kernel.device


        if method not in ["srht_2", "srht_3", "srht"]:
            raise ValueError("Unknown method supplied for tuning preconditioner "
                    "construction.")

        if method.startswith("srht_"):
            n_passes = int(method.split("_")[1])
            self.u_mat, self.eig, self.z_trans_y, self.y_trans_y = \
                            initialize_srht_multipass(dataset,
                                max_rank, kernel, random_state, verbose,
                                n_passes, get_zty = True)
        elif method == "srht":
            self.u_mat, self.eig, self.z_trans_y, self.y_trans_y = \
                            initialize_srht(dataset, max_rank,
                                kernel, random_state, verbose, get_zty = True)

        lambda_ = kernel.get_lambda()
        min_eig = self.eig.min()
        self.eig += lambda_**2

        self.inv_eig = self.eig.copy()
        mask = self.inv_eig > 1e-14
        self.inv_eig[mask] = 1 / self.inv_eig[mask]
        self.inv_eig[mask==False] = 0.0

        self.achieved_ratio = min_eig / lambda_**2
        self.prefactor = float(min_eig + lambda_**2)


    def batch_matvec(self, xvec):
        """Returns a matvec of the preconditioner with a set of input
        vectors xvec."""
        xprod = self.u_mat.T @ xvec
        xprod1 = self.u_mat @ (self.inv_eig[:,None] * self.prefactor * xprod)
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
        preconditioner construction."""
        return self.z_trans_y


    def get_yty(self):
        """Returns the product y^T y, which is assembled during
        preconditioner construction."""
        return float(self.y_trans_y)
