"""Checks the fast Hadamard transform based polynomial kernel
feature generation routines to ensure they are producing
correct results by comparing with a clunky, simple
mostly Python implementation."""
from math import ceil
import sys

import unittest
import numpy as np
from scipy.stats import chi

from cpu_rf_gen_module import cpuFastHadamardTransform as cFHT
from cpu_rf_gen_module import cpuGraphPolyFHT
from cpu_rf_gen_module import cpuPolyFHT

try:
    from cuda_rf_gen_module import gpuGraphPolyFHT
    from cuda_rf_gen_module import gpuPolyFHT
    import cupy as cp
except:
    pass

class TestPolyFHT(unittest.TestCase):
    """Tests feature generation for the polynomial kernels."""

    def test_graph_poly(self):
        """Tests the graph polynomial kernel."""
        num_feats = 56
        num_freqs = 1000
        ndatapoints = 124

        outcomes = run_evaluation(ndatapoints, num_feats,
                    num_freqs, precision = "double",
                    kernel_type = "graph")
        for outcome in outcomes:
            self.assertTrue(outcome)

        outcomes = run_evaluation(ndatapoints, num_feats,
                    num_freqs, precision = "float",
                    kernel_type = "graph")
        for outcome in outcomes:
            self.assertTrue(outcome)

        num_feats = 512
        num_freqs = 556
        ndatapoints = 2000
        outcomes = run_evaluation(ndatapoints, num_feats,
                    num_freqs, precision = "double",
                    kernel_type = "graph")
        for outcome in outcomes:
            self.assertTrue(outcome)


    def test_classic_poly(self):
        """Tests the classic polynomial kernel."""
        num_feats = 56
        num_freqs = 1000
        ndatapoints = 124

        outcomes = run_evaluation(ndatapoints, num_feats,
                    num_freqs, precision = "double",
                    kernel_type = "classic")
        for outcome in outcomes:
            self.assertTrue(outcome)

        outcomes = run_evaluation(ndatapoints, num_feats,
                    num_freqs, precision = "float",
                    kernel_type = "classic")
        for outcome in outcomes:
            self.assertTrue(outcome)

        num_feats = 512
        num_freqs = 556
        ndatapoints = 2000
        outcomes = run_evaluation(ndatapoints, num_feats,
                    num_freqs, precision = "double",
                    kernel_type = "classic")
        for outcome in outcomes:
            self.assertTrue(outcome)



def run_evaluation(ndatapoints, num_feats, num_freqs,
        polydegree = 2, num_threads = 2, precision = "double",
        kernel_type = "graph"):
    """Runs an evaluation using some set of input settings
    (e.g. use CPU or GPU, use X number of frequencies etc."""
    if kernel_type == "classic":
        xdata, col_sampler, radem, features = get_classic_matrices(ndatapoints,
                        num_feats, num_freqs, polydegree, precision)
        true_features = get_classic_features(xdata, radem, col_sampler,
                precision, num_freqs)
        conv_fun = cpuPolyFHT
    else:
        xdata, col_sampler, radem, features, xdim2 = get_graph_matrices(ndatapoints,
                        num_feats, num_freqs, polydegree, precision)
        true_features = get_graph_features(xdata, radem, col_sampler, xdim2, precision)
        conv_fun = cpuGraphPolyFHT

    if kernel_type == "classic":
        conv_fun(xdata, radem, col_sampler, features,
                polydegree, num_threads, num_freqs)
    else:
        conv_fun(xdata, radem, col_sampler, features,
                polydegree, num_threads)

    if precision == "float":
        outcome = np.allclose(features, true_features, rtol=1e-4, atol=1e-4)
    else:
        outcome = np.allclose(features, true_features)
    print(f"Does result match for ndatapoints={ndatapoints}, "
            f"num_aas={num_feats}, polydegree={polydegree},"
            f" kernel {kernel_type}?\n{outcome}")

    if "cupy" not in sys.modules:
        return [outcome]

    if kernel_type == "graph":
        conv_fun = gpuGraphPolyFHT
    else:
        conv_fun = gpuPolyFHT

    col_sampler = cp.asarray(col_sampler)
    xold = xdata.copy()
    xdata = cp.asarray(xdata)
    features = cp.asarray(features)
    radem = cp.asarray(radem)
    if kernel_type == "classic":
        conv_fun(xdata, radem, col_sampler, features,
            polydegree, num_threads, num_freqs)
    else:
        conv_fun(xdata, radem, col_sampler, features,
            polydegree, num_threads)

    features = cp.asnumpy(features)
    if precision == "float":
        outcome_cuda = np.allclose(features, true_features, rtol=1e-4, atol=1e-4)
    else:
        outcome_cuda = np.allclose(features, true_features)
    print(f"Does result ON CUDA match for ndatapoints={ndatapoints}, "
            f"num_aas={num_feats}, polydegree={polydegree},"
            f"kernel {kernel_type}?\n{outcome_cuda}")
    return outcome, outcome_cuda



def get_graph_matrices(ndatapoints, num_feats, num_freqs, polydegree,
        precision = "double"):
    """Gets all of the matrices that would ordinarily be generated by
    the graph poly kernel for purposes of running the test."""
    padded_dims = 2**ceil(np.log2(max(num_feats + 1, 2)))
    num_repeats = ceil(num_freqs / padded_dims)
    init_calc_freqsize = num_repeats * padded_dims

    rng = np.random.default_rng(123)
    x_data = rng.uniform(size=(ndatapoints, 9, num_feats))
    padded_x = np.zeros((ndatapoints, 9, padded_dims))
    padded_x[:,:,:num_feats] = x_data

    S = chi.rvs(df=padded_dims,
            size=(polydegree, init_calc_freqsize),
            random_state = 123)

    radem_array = np.asarray([-1,1], dtype=np.int8)
    radem_diag = rng.choice(radem_array, size=(3 * polydegree, 1,
                    init_calc_freqsize), replace=True)
    features = np.zeros((ndatapoints, init_calc_freqsize))
    if precision == "float":
        return padded_x.astype(np.float32), S.astype(np.float32),\
                radem_diag, features.astype(np.float32), init_calc_freqsize
    return padded_x, S, radem_diag, features, init_calc_freqsize


def get_classic_matrices(ndatapoints, num_feats, num_freqs, polydegree,
        precision = "double"):
    """Gets all of the matrices that would ordinarily be generated by
    the poly kernel for purposes of running the test."""
    padded_dims = 2**ceil(np.log2(max(num_feats, 2)))
    nblocks = ceil(num_freqs / padded_dims)

    rng = np.random.default_rng(123)
    x_data = rng.uniform(size=(ndatapoints, nblocks, num_feats))
    padded_x = np.zeros((ndatapoints, nblocks, padded_dims))
    padded_x[:,:,:num_feats] = x_data

    S = chi.rvs(df=padded_dims,
            size=(polydegree, nblocks, padded_dims),
            random_state = 123)

    radem_array = np.asarray([-1,1], dtype=np.int8)
    radem_diag = rng.choice(radem_array, size=(5 * polydegree, nblocks,
                    padded_dims), replace=True)
    features = np.zeros((ndatapoints, num_freqs))
    if precision == "float":
        return padded_x.astype(np.float32), S.astype(np.float32),\
                radem_diag, features
    return padded_x, S, radem_diag, features


def get_graph_features(xdata, radem, S, init_calc_freqsize,
        precision = "double"):
    """A very slow and clunky python version of the feature calculation.
    Intended to double-check the Cython version and make sure it is correct.
    This function is for the graph poly feature generation."""
    true_features = np.zeros((xdata.shape[0], init_calc_freqsize))
    num_repeats = ceil(init_calc_freqsize / xdata.shape[2])
    fht_func = cFHT
    if precision == "float":
        true_features = true_features.astype(np.float32)

    start, cutoff = 0, xdata.shape[2]
    norm_constant = np.log2(xdata.shape[2]) / 2
    norm_constant = 1 / (2**norm_constant)
    for i in range(num_repeats):
        pre_sum_feats = xdata * radem[0:1,0:1,start:cutoff] * norm_constant
        fht_func(pre_sum_feats, 2)
        pre_sum_feats *= radem[1:2,0:1,start:cutoff] * norm_constant
        fht_func(pre_sum_feats, 2)
        pre_sum_feats *= radem[2:3,0:1,start:cutoff] * norm_constant
        fht_func(pre_sum_feats, 2)
        pre_sum_feats *= S[0:1,None,start:cutoff]

        for j in range(1, S.shape[0]):
            x_updated = xdata * radem[3*j,0,start:cutoff][None,None,:] * norm_constant
            fht_func(x_updated, 2)
            x_updated *= radem[3*j+1,0,start:cutoff][None,None,:] * norm_constant
            fht_func(x_updated, 2)
            x_updated *= radem[3*j+2,0,start:cutoff][None,None,:] * norm_constant
            fht_func(x_updated, 2)

            x_updated *= S[j:j+1,None,start:cutoff]
            pre_sum_feats *= x_updated

        true_features[:,start:cutoff] = pre_sum_feats.sum(axis=1)
        start += xdata.shape[2]
        cutoff += xdata.shape[2]
    return true_features


def get_classic_features(xdata, radem, S, precision = "double",
        num_freqs = 24):
    """A very slow and clunky python version of the feature calculation.
    Intended to double-check the Cython version and make sure it is correct.
    This function is for the classic poly feature generation."""
    norm_constant = np.log2(xdata.shape[2]) / 2
    norm_constant = 1 / (2**norm_constant)

    true_features = xdata * radem[0:1,:,:] * norm_constant
    cFHT(true_features, 2)
    true_features *= radem[1:2,:,:] * norm_constant
    cFHT(true_features, 2)
    true_features *= radem[2:3,:,:] * norm_constant
    cFHT(true_features, 2)
    true_features *= radem[3:4,:,:] * norm_constant
    cFHT(true_features, 2)
    true_features *= radem[4:5,:,:] * norm_constant
    cFHT(true_features, 2)
    true_features *= S[0,:,:]

    for j in range(1, S.shape[0]):
        x_updated = xdata * radem[5*j,:,:] * norm_constant
        cFHT(x_updated, 2)
        x_updated *= radem[5*j+1,:,:] * norm_constant
        cFHT(x_updated, 2)
        x_updated *= radem[5*j+2,:,:] * norm_constant
        cFHT(x_updated, 2)
        x_updated *= radem[5*j+3,:,:] * norm_constant
        cFHT(x_updated, 2)
        x_updated *= radem[5*j+4,:,:] * norm_constant
        cFHT(x_updated, 2)
        x_updated *= S[j,:,:]
        true_features *= x_updated

    true_features = true_features.reshape((true_features.shape[0],
        true_features.shape[1] * true_features.shape[2]))[:,:num_freqs]

    return true_features


if __name__ == "__main__":
    unittest.main()
