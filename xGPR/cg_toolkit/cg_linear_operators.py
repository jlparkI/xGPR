"""Linear operators for performing matvecs for conjugate gradients,
one for CPU, one for CUDA."""
from scipy.sparse.linalg import LinearOperator
import numpy as np
try:
    import cupy as cp
    from cupyx.scipy.sparse.linalg import LinearOperator as cpx_LinearOperator
except:
    pass

class Cuda_CGLinearOperator(cpx_LinearOperator):
    """A Cuda linear operator for evaluating (Z^T Z + lambda) vec
    for conjugate gradients, where Z is the random features
    generated for the input dataset.

    Attributes:
        n_iter (int): The total number of CG iterations.
        dataset: An OnlineDataset or OfflineDataset containing
            the raw data.
        kernel: A valid kernel object that can generate random
            features given a chunk of raw input data.
        verbose (bool): If True, print regular updates.
        cpu_override (bool): If True, if the data has been
            pretransformed (i.e. random features pregenerated)
            and device is Cuda, perform matvecs on CPU rather
            than GPU to avoid the cost of transferring feature
            data to GPU.
    """

    def __init__(self, dataset, kernel, verbose, cpu_override = False):
        """Class constructor.

        Args:
            dataset: An OnlineDataset or OfflineDataset containing
                raw data.
            kernel: A valid kernel object that can generate random
                features for a chunk of input data.
            verbose (bool): If True, print regular updates.
            cpu_override (bool): If True, if the data has been
                pretransformed (i.e. random features pregenerated)
                and device is Cuda, perform matvecs on CPU rather
                than GPU to avoid the cost of transferring feature
                data to GPU.
        """
        super().__init__(shape=(kernel.get_num_rffs(),
                            kernel.get_num_rffs()),
                            dtype=cp.float64)
        self.n_iter = 0
        self.kernel = kernel
        self.dataset = dataset
        self.verbose = verbose
        self.cpu_override = cpu_override

    def _matvec(self, x):
        """Calculates (Z^T Z + lambda) vec, where Z is
        the random features generated for self.dataset.

        Args:
            x (cp.ndarray): The vector against which
                to multiply.

        Returns:
            xprod (cp.ndarray): the matvec product.
        """
        if self.verbose and self.n_iter % 5 == 0:
            print(f"Iteration {self.n_iter}")
        self.n_iter += 1
        xprod = self.kernel.get_lambda()**2 * x
        if self.dataset.pretransformed and self.cpu_override:
            #Sending the data to GPU (depending on hardware) takes
            #longer in most cases than just performing the matvec
            #on CPU, if there is no other GPU work to perform.
            self.dataset.device = "cpu"
            x_cpu = cp.asnumpy(x)
            x_prod_cpu = cp.asnumpy(xprod)
            for xdata in self.dataset.get_chunked_x_data():
                x_prod_cpu += (xdata.T @ (xdata @ x_cpu))
            self.dataset.device = "gpu"
            xprod = cp.asarray(x_prod_cpu)

        elif self.dataset.pretransformed:
            for xdata in self.dataset.get_chunked_x_data():
                xprod += (xdata.T @ (xdata @ x))

        else:
            for xdata in self.dataset.get_chunked_x_data():
                xdata = self.kernel.transform_x(xdata)
                xprod += (xdata.T @ (xdata @ x))
        return xprod


class CPU_CGLinearOperator(LinearOperator):
    """A CPU linear operator for evaluating (Z^T Z + lambda) vec
    for conjugate gradients, where Z is the random features
    generated for the input dataset.

    Attributes:
        n_iter (int): The total number of CG iterations.
        dataset: An OnlineDataset or OfflineDataset containing
            the raw data.
        kernel: A valid kernel object that can generate random
            features given a chunk of raw input data.
        verbose (bool): If True, print regular updates.
    """

    def __init__(self, dataset, kernel, verbose):
        """Class constructor.

        Args:
            dataset: An OnlineDataset or OfflineDataset containing
                raw data.
            kernel: A valid kernel object that can generate random
                features for a chunk of input data.
            verbose (bool): If True, print regular updates.
        """
        super().__init__(shape=(kernel.get_num_rffs(),
                            kernel.get_num_rffs()),
                            dtype=np.float64)
        self.n_iter = 0
        self.kernel = kernel
        self.dataset = dataset
        self.verbose = verbose


    def _matvec(self, x):
        """Calculates (Z^T Z + lambda) vec, where Z is
        the random features generated for self.dataset.

        Args:
            x (cp.ndarray): The vector against which
                to multiply.

        Returns:
            xprod (cp.ndarray): the matvec product.
        """
        if self.verbose and self.n_iter % 5 == 0:
            print(f"Iteration {self.n_iter}")
        self.n_iter += 1
        xprod = self.kernel.get_lambda()**2 * x
        if self.dataset.pretransformed:
            for xdata in self.dataset.get_chunked_x_data():
                xprod += (xdata.T @ (xdata @ x))
        else:
            for xdata in self.dataset.get_chunked_x_data():
                xdata = self.kernel.transform_x(xdata)
                xprod += (xdata.T @ (xdata @ x))
        return xprod
