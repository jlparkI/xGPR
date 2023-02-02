"""Implements a randomized nystrom preconditioner for CPU and Cuda."""
from scipy.sparse.linalg import LinearOperator
import numpy as np
try:
    import cupy as cp
    from cupyx.scipy.sparse.linalg import LinearOperator as cpx_LinearOperator
except:
    pass
from .rand_nys_constructors import initialize_gauss, initialize_srht, initialize_srht_multipass



class Cuda_RandNysPreconditioner(cpx_LinearOperator):
    """Implements a preconditioner based on the randomized
    Nystrom approximation to the inverse of Z^T Z + lambda;
    for Cuda only.

    Attributes:
        achieved_ratio (float): lambda_ / min_eig, where min_eig is the
            smallest eigenvalue for the preconditioner. This is a
            reasonably good predictor of how effective the preconditioner
            will be. Smaller is better.
        prefactor (float): A constant by which the matvec is multiplied
            when performing matvecs with the preconditioner.
        u_mat (cp.ndarray): A cupy array containing the eigenvectors of
            the matrix formed during preconditioner construction. Used
            together with prefactor and inv_eig to perform the matvec
            that approximates A^-1.
        inv_eig (cp.ndarray): A cupy array containing the inverse of the
            eigenvalues of the matrix formed during preconditioner construction.
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
            method (str): One of "srht", "gauss", "srht_2". Determines the method of
                preconditioner construction.
        """
        super().__init__(shape=(kernel.get_num_rffs(),
                            kernel.get_num_rffs()),
                            dtype=cp.float64)
        if method not in ["srht_2", "srht_3", "srht", "gauss"]:
            raise ValueError("Unknown method supplied for preconditioner "
                    "construction.")

        if method.startswith("srht_"):
            n_passes = int(method.split("_")[1])
            self.u_mat, s_mat, _, _ = initialize_srht_multipass(dataset, max_rank,
                                kernel, random_state, verbose, n_passes)
        elif method == "srht":
            self.u_mat, s_mat, _, _ = initialize_srht(dataset, max_rank,
                                kernel, random_state, verbose)
        else:
            self.u_mat, s_mat, _, _ = initialize_gauss(dataset, max_rank,
                                kernel, random_state, verbose)

        lambda_ = kernel.get_lambda()

        self.inv_eig = s_mat + lambda_**2
        mask = self.inv_eig > 1e-14
        self.inv_eig[mask] = 1 / self.inv_eig[mask]
        self.inv_eig[mask==False] = 0.0

        min_eig = s_mat.min()
        self.achieved_ratio = min_eig / lambda_**2
        self.prefactor = float(min_eig + lambda_**2)


    def batch_matvec(self, xvec):
        """Returns a matvec of the preconditioner with a set of input
        vectors xvec."""
        xprod = self.u_mat.T @ xvec
        xprod1 = self.u_mat @ (self.inv_eig[:,None] * self.prefactor * xprod)
        xprod2 = xvec - (self.u_mat @ xprod)
        return xprod2 + xprod1


    def _matvec(self, xvec):
        """Implements the matvec for a single input vector."""
        xprod = self.u_mat.T @ xvec
        xprod1 = self.u_mat @ (self.inv_eig * self.prefactor * xprod)
        xprod2 = xvec - (self.u_mat @ xprod)
        return xprod2 + xprod1




class CPU_RandNysPreconditioner(LinearOperator):
    """Implements a preconditioner based on the randomized
    Nystrom approximation to the inverse of Z^T Z + lambda;
    for CPU only.

    Attributes:
        achieved_ratio (float): lambda_ / min_eig, where min_eig is the
            smallest eigenvalue for the preconditioner. This is a
            reasonably good predictor of how effective the preconditioner
            will be. Smaller is better.
        prefactor (float): A constant by which the matvec is multiplied
            when performing matvecs with the preconditioner.
        u_mat (cp.ndarray): A cupy array containing the eigenvectors of
            the matrix formed during preconditioner construction. Used
            together with prefactor and inv_eig to perform the matvec
            that approximates A^-1.
        inv_eig (cp.ndarray): A cupy array containing the inverse of the
            eigenvalues of the matrix formed during preconditioner construction.
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
            method (str): One of "srht", "gauss", "srht_2". Determines the method of
                preconditioner construction.
        """
        super().__init__(shape=(kernel.get_num_rffs(),
                            kernel.get_num_rffs()),
                            dtype=cp.float64)
        if method not in ["srht_2", "srht_3", "srht", "gauss"]:
            raise ValueError("Unknown method supplied for preconditioner "
                    "construction.")

        if method.startswith("srht_"):
            n_passes = int(method.split("_")[1])
            self.u_mat, s_mat, _, _ = initialize_srht_multipass(dataset, max_rank,
                                kernel, random_state, verbose, n_passes)
        elif method == "srht":
            self.u_mat, s_mat, _, _ = initialize_srht(dataset, max_rank,
                                kernel, random_state, verbose)
        else:
            self.u_mat, s_mat, _, _ = initialize_gauss(dataset, max_rank,
                                kernel, random_state, verbose)

        lambda_ = kernel.get_lambda()

        self.inv_eig = s_mat + lambda_**2
        mask = self.inv_eig > 1e-14
        self.inv_eig[mask] = 1 / self.inv_eig[mask]
        self.inv_eig[mask==False] = 0.0

        min_eig = s_mat.min()
        self.achieved_ratio = min_eig / lambda_**2
        self.prefactor = float(min_eig + lambda_**2)


    def batch_matvec(self, xvec):
        """Returns a matvec of the preconditioner with a set of input
        vectors xvec."""
        xprod = self.u_mat.T @ xvec
        xprod1 = self.u_mat @ (self.inv_eig[:,None] * self.prefactor * xprod)
        xprod2 = xvec - (self.u_mat @ xprod)
        return xprod2 + xprod1


    def _matvec(self, xvec):
        """Implements the matvec for a single input vector."""
        xprod = self.u_mat.T @ xvec
        xprod1 = self.u_mat @ (self.inv_eig * self.prefactor * xprod)
        xprod2 = xvec - (self.u_mat @ xprod)
        return xprod2 + xprod1
