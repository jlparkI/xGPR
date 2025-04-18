"""A quick and dirty timeit script for checking the time to perform graph
and polynomial feature generation."""
import math
import timeit
import numpy as np
import cupy as cp
import cupyx
from scipy.stats import chi
from xGPR.xgpr_cpu_rfgen_cpp_ext import cpuConv1dFGen
from xGPR.xgpr_cuda_rfgen_cpp_ext import cudaConv1dFGen as gpuConv1dFGen


#Running the CPU numpy test with too many columns is extremely
#expensive -- avoid running that test by accident.
MAX_COLS_MATMUL = 124


def timetest():
    """Run a series of timeit tests and print the results."""
    #Change these lines depending on what dims you want to test.
    nrows, ncols, nnodes, conv_width = 256, 100, 70, 9

    ntests = 100
    print("**************************************")
    print(f"Running test with {nrows} rows, {ncols} cols, {nnodes} nodes.")
    print("**************************************")


    cpu_graph_setup = f"""
import math
import numpy as np
from scipy.stats import chi
from xGPR.xgpr_cpu_rfgen_cpp_ext import cpuConv1dFGen
random_seed = 123
rng = np.random.default_rng(random_seed)
marr = rng.uniform(low=-10.0, high=10.0, size=({nrows},{nnodes},{ncols}))
marr = marr.astype(np.float32)
radem_array = np.asarray([-1.0, 1.0])
padded_width = 2**math.ceil(np.log2(max({ncols * conv_width}, 2)))
D1 = rng.choice(radem_array, size=(3,1,padded_width), replace=True)
D1 = D1.astype(np.int8)
chi_arr = chi.rvs(df={ncols}, size=padded_width, random_state=random_seed)
chi_arr = chi_arr.astype(np.float32)
from __main__ import graphconv_cpu_test"""
    print(f"Time(us) for GRAPH cpu:")
    time_taken = timeit.timeit(f"graphconv_cpu_test(marr, D1, chi_arr, {conv_width})",
            setup=cpu_graph_setup, number=ntests)
    print(1e6 * time_taken / ntests)


    random_seed = 123
    rng = np.random.default_rng(random_seed)
    marr = rng.uniform(low=-10.0, high=10.0, size=(nrows,nnodes,ncols))
    marr = marr.astype(np.float32)
    radem_array = np.asarray([-1.0, 1.0])
    padded_width = 2**math.ceil(np.log2(max(ncols * conv_width, 2)))
    D1 = rng.choice(radem_array, size=(3,1,padded_width), replace=True)
    D1 = D1.astype(np.int8)
    chi_arr = chi.rvs(df=ncols, size=padded_width, random_state=random_seed)
    chi_arr = chi_arr.astype(np.float32)
    marr, D1, chi_arr = cp.asarray(marr), cp.asarray(D1), cp.asarray(chi_arr)
    seqlen = np.full(marr.shape[0], marr.shape[1], dtype=np.int32)
    print(f"Time(us) for GRAPH gpu:")
    time_taken = cupyx.time.repeat(graphconv_gpu_test,
            (marr, D1, chi_arr, conv_width, seqlen),
            n_repeat = ntests)
    print(time_taken)





def graphconv_cpu_test(marr, diag, chi_arr, conv_width):
    """Run the CPU graph convolution feature generation routine."""
    output_arr = np.zeros((marr.shape[0], diag.shape[2] * 2))
    seqlen = np.full(marr.shape[0], marr.shape[1]).astype(np.int32)
    cpuConv1dFGen(marr, output_arr, diag, chi_arr, seqlen, conv_width,
            0)


def graphconv_gpu_test(marr, diag, chi_arr, conv_width, seqlen):
    """Run the CPU graph convolution feature generation routine."""
    output_arr = cp.zeros((marr.shape[0], diag.shape[2] * 2))
    gpuConv1dFGen(marr, output_arr, diag, chi_arr, seqlen, conv_width,
            0)


if __name__ == "__main__":
    timetest()
