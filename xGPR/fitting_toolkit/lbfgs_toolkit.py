"""Contains tools for fitting a classification model via L-BFGS."""
import numpy as np
try:
    import cupy as cp
except:
    pass




def lbfgs_cost_fun(params, dataset, kernel, nclasses, gamma):
    """Evaluates the cost and gradient for L-BFGS for
    a classification problem."""
    num_rffs = kernel.get_num_rffs()
    weights = np.zeros((num_rffs, nclasses))
    nonflat_gradient = kernel.get_lambda()**2 * weights
    nonflat_gradient[0,:] = 0
    cuda = False

    for k, i in enumerate(range(0, params.shape[0], num_rffs)):
        weights[:,k] = params[i:i+num_rffs]

    loss = 0.5 * kernel.get_lambda()**2 * (weights**2)[1:,:].sum()

    if kernel.device == "cuda":
        weights = cp.asarray(weights)
        nonflat_gradient = cp.asarray(nonflat_gradient)
        cuda = True

    for (xdata, ydata, ldata) in dataset.get_chunked_data():
        xd, yd = kernel.transform_x_y(xdata, ydata, ldata,
                classification=True)
        pred = xd @ weights + gamma[None,:]
        # Numerically stable softmax.
        pred -= pred.max(axis=1)[:,None]
        pred = 2.71828**pred
        pred /= pred.sum(axis=1)[:,None]
        if cuda:
            logpred = cp.log(pred.clip(min=1e-16))
        else:
            logpred = np.log(pred.clip(min=1e-16))
        loss -= float(logpred[np.arange(pred.shape[0]), yd].sum(axis=0))

        for k in range(nclasses):
            if cuda:
                flarr = (yd==k).astype(cp.float64)
            else:
                flarr = (yd==k).astype(np.float64)
            nonflat_gradient[:,k] += ((pred[:,k] - flarr)[:,None] * xd).sum(axis=0)

    if cuda:
        nonflat_gradient = cp.asnumpy(nonflat_gradient)

    flat_gradient = np.zeros((num_rffs * nclasses))

    for k, i in enumerate(range(0, flat_gradient.shape[0], num_rffs)):
        flat_gradient[i:i+num_rffs] = nonflat_gradient[:,k]

    print(f"Function eval complete. Loss {loss}", flush=True)

    return loss, flat_gradient
