"""Implements a randomized nystrom preconditioner that can
be switched back and forth between Cuda and CPU. This is not
possible for the other variants since they have to inherit from
either Scipy or Cupy.scipy.sparse."""
from scipy.sparse.linalg import LinearOperator
import numpy as np
try:
    import cupy as cp
except:
    pass
from .rand_nys_constructors import initialize_gauss, initialize_srht, initialize_srht_multipass



class InterDevicePreconditioner():
    """Implements a preconditioner based on the randomized
    Nystrom approximation to the inverse of Z^T Z + lambda;
    unlike the other preconditioners, this one can be switched
    back and forth between devices.

    Attributes:
        achieved_ratio (float): lambda_ / min_eig, where min_eig is the
            smallest eigenvalue for the preconditioner. This is a
            reasonably good predictor of how effective the preconditioner
            will be. Smaller is better.
        prefactor (float): A constant by which the matvec is multiplied
            when performing matvecs with the preconditioner.
        u_mat (array): A cupy array containing the eigenvectors of
            the matrix formed during preconditioner construction. Used
            together with prefactor and inv_eig to perform the matvec
            that approximates A^-1.
        inv_eig (array): A cupy array containing the inverse of the
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
        self.achieved_ratio = float(min_eig / lambda_**2)
        self.prefactor = float(min_eig + lambda_**2)

        self.device = kernel.device


    def batch_matvec(self, xvec):
        """Returns a matvec of the preconditioner with a set of input
        vectors xvec."""
        xprod = self.u_mat.T @ xvec
        xprod1 = self.u_mat @ (self.inv_eig[:,None] * self.prefactor * xprod)
        xprod2 = xvec - (self.u_mat @ xprod)
        return xprod2 + xprod1


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
            value (str): Must be one of 'cpu', 'gpu'.

        Raises:
            ValueError: A ValueError is raised if an unrecognized
                device is passed.
        """
        if value == "cpu":
            if not isinstance(self.u_mat, np.ndarray):
                self.u_mat = cp.asnumpy(self.u_mat)
                self.inv_eig = cp.asnumpy(self.inv_eig)

        elif value == "gpu":
            self.u_mat = cp.asarray(self.u_mat)
            self.inv_eig = cp.asarray(self.inv_eig)

        else:
            raise ValueError("Unrecognized device supplied. Must be one "
                    "of 'cpu', 'gpu'.")

        self.device_ = value
