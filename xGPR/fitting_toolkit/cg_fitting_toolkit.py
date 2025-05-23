"""Implements functions for fitting (once hyperparameters have been
selected) using CG, either using our internal routine or Scipy / Cupy's."""
import warnings
import numpy as np
from .cg_tools import CPU_ConjugateGrad

try:
    import cupy as cp
    from .cg_tools import GPU_ConjugateGrad
except:
    pass

from ..scoring_toolkit.exact_nmll_calcs import calc_zty




def cg_fit_lib_internal(kernel, dataset, cg_tol = 1e-4, max_iter = 500,
                        preconditioner = None, verbose = True):
    """Calculates the weights when fitting the model using
    preconditioned CG. Good scaling but slower for small
    numbers of random features.

    Args:
        kernel: A valid kernel object that can generate random features.
        dataset: Either OnlineDataset or OfflineDataset.
        cg_tol (float): The threshold below which cg is deemed to have
            converged. Defaults to 1e-5.
        max_iter (int): The maximum number of iterations before
            CG is deemed to have failed to converge.
        preconditioner: Either None or a valid Preconditioner (e.g.
            CudaRandomizedPreconditioner, CPURandomizedPreconditioner
            etc). If None, no preconditioning is used. Otherwise,
            the preconditioner is used for CG. The preconditioner
            can be built by calling self.build_preconditioner
            with appropriate arguments.
        verbose (bool): If True, print regular updates.

    Returns:
        weights: A cupy or numpy array of shape (M) for M
            random features.
        n_iter (int): The number of CG iterations.
        losses (list): The loss on each iteration; for diagnostic
            purposes.
    """
    if kernel.device == "cuda":
        cg_operator = GPU_ConjugateGrad()
        resid = cp.zeros((kernel.get_num_rffs(), 2, 1))
    else:
        cg_operator = CPU_ConjugateGrad()
        resid = np.zeros((kernel.get_num_rffs(), 2, 1))

    if preconditioner is None:
        z_trans_y, _ = calc_zty(dataset, kernel)
    else:
        z_trans_y = preconditioner.get_zty()

    resid[:,0,:] = z_trans_y[:,None] / dataset.get_ndatapoints()

    weights, converged, n_iter, losses = cg_operator.fit(dataset, kernel,
                preconditioner, resid, max_iter, cg_tol, verbose,
                nmll_settings = False)
    weights *= dataset.get_ndatapoints()
    if not converged:
        warnings.warn("Conjugate gradients failed to converge! Try refitting "
                        "the model with updated settings.")

    if verbose:
        print(f"CG iterations: {n_iter}")
    return weights, n_iter, losses
