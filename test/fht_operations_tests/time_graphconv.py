"""A quick and dirty timeit script for checking the time to perform graph
and polynomial feature generation."""
import timeit
import numpy as np
import cupy as cp
import cupyx
from scipy.stats import chi
from cpu_convolution_float_hadamard_operations import floatCpuConv1dFGen, floatCpuGraphPolyFHT
from cuda_convolution_float_hadamard_operations import floatGpuConv1dFGen, floatGpuGraphPolyFHT


#Running the CPU numpy test with too many columns is extremely
#expensive -- avoid running that test by accident.
MAX_COLS_MATMUL = 124


def timetest():
    """Run a series of timeit tests and print the results."""
    #Change these lines depending on what dims you want to test.
    nthreads = 2
    nrows, ncols, nnodes = 256, 512, 62

    ntests = 100
    print("**************************************")
    print(f"Running test with {nrows} rows, {ncols} cols, {nnodes} nodes.")
    print("**************************************")


    cpu_graph_setup = f"""
nthreads = {nthreads}
import math
import numpy as np
from scipy.stats import chi
from cpu_convolution_float_hadamard_operations import floatCpuConv1dFGen
random_seed = 123
rng = np.random.default_rng(random_seed)
marr = rng.uniform(low=-10.0, high=10.0, size=({nrows},{nnodes},{ncols}))
marr = marr.astype(np.float32)
radem_array = np.asarray([-1.0, 1.0])
D1 = rng.choice(radem_array, size=(3,{1},{ncols}), replace=True)
D1 = D1.astype(np.int8)
chi_arr = chi.rvs(df={ncols}, size={ncols}, random_state=random_seed)
chi_arr = chi_arr.astype(np.float32)
from __main__ import graphconv_cpu_test"""
    print(f"Time(us) for GRAPH cpu with {nthreads} threads:")
    time_taken = timeit.timeit("graphconv_cpu_test(marr, D1, chi_arr, nthreads)",
            setup=cpu_graph_setup, number=ntests)
    print(1e6 * time_taken / ntests)
    
    
    random_seed = 123
    rng = np.random.default_rng(random_seed)
    marr = rng.uniform(low=-10.0, high=10.0, size=(nrows,nnodes,ncols))
    marr = marr.astype(np.float32)
    radem_array = np.asarray([-1.0, 1.0])
    D1 = rng.choice(radem_array, size=(3,1,ncols), replace=True)
    D1 = D1.astype(np.int8)
    chi_arr = chi.rvs(df=ncols, size=ncols, random_state=random_seed)
    chi_arr = chi_arr.astype(np.float32)
    marr, D1, chi_arr = cp.asarray(marr), cp.asarray(D1), cp.asarray(chi_arr)
    print(f"Time(us) for GRAPH gpu with {nthreads} threads:")
    time_taken = cupyx.time.repeat(graphconv_gpu_test, (marr, D1, chi_arr), n_repeat = ntests)
    print(time_taken)


    
    cpu_poly_setup = f"""
nthreads = {nthreads}
import math
import numpy as np
from scipy.stats import chi
from cpu_convolution_float_hadamard_operations import floatCpuGraphPolyFHT
random_seed = 123
rng = np.random.default_rng(random_seed)
marr = rng.uniform(low=-10.0, high=10.0, size=({nrows},{nnodes},{ncols}))
marr = marr.astype(np.float32)
radem_array = np.asarray([-1.0, 1.0])
D1 = rng.choice(radem_array, size=(6,{1},{ncols}), replace=True)
D1 = D1.astype(np.int8)
chi_arr = chi.rvs(df={ncols}, size=(2,{ncols}), random_state=random_seed)
chi_arr = chi_arr.astype(np.float32)
from __main__ import poly_cpu_test"""
    print(f"Time(us) for POLY cpu with {nthreads} threads:")
    time_taken = timeit.timeit("poly_cpu_test(marr, D1, chi_arr, nthreads)",
            setup=cpu_poly_setup, number=ntests)
    print(1e6 * time_taken / ntests)


    random_seed = 123
    rng = np.random.default_rng(random_seed)
    marr = rng.uniform(low=-10.0, high=10.0, size=(nrows,nnodes,ncols))
    marr = marr.astype(np.float32)
    radem_array = np.asarray([-1.0, 1.0])
    D1 = rng.choice(radem_array, size=(6,1,ncols), replace=True)
    D1 = D1.astype(np.int8)
    chi_arr = chi.rvs(df=ncols, size=(2,ncols), random_state=random_seed)
    chi_arr = chi_arr.astype(np.float32)
    marr, D1, chi_arr = cp.asarray(marr), cp.asarray(D1), cp.asarray(chi_arr)
    print(f"Time(us) for POLY gpu with {nthreads} threads:")
    time_taken = cupyx.time.repeat(poly_gpu_test, (marr, D1, chi_arr), n_repeat = ntests)
    print(time_taken)



def graphconv_cpu_test(marr, diag, chi_arr, nthreads):
    """Run the CPU graph convolution feature generation routine."""
    output_arr = np.zeros((marr.shape[0], diag.shape[2] * 2))
    beta = 1.0
    floatCpuConv1dFGen(marr, diag, output_arr, chi_arr, nthreads,
                        beta)


def graphconv_gpu_test(marr, diag, chi_arr):
    """Run the CPU graph convolution feature generation routine."""
    output_arr = cp.zeros((marr.shape[0], diag.shape[2] * 2))
    beta = 1.0
    floatGpuConv1dFGen(marr, diag, output_arr, chi_arr, 2, beta)

def poly_cpu_test(marr, diag, chi_arr, nthreads):
    output_arr = np.zeros((marr.shape[0], diag.shape[2]), dtype=np.float32)
    floatCpuGraphPolyFHT(marr, diag, chi_arr, output_arr, 2, nthreads)


def poly_gpu_test(marr, diag, chi_arr):
    output_arr = cp.zeros((marr.shape[0], diag.shape[2]), dtype=np.float32)
    floatGpuGraphPolyFHT(marr, diag, chi_arr, output_arr, 2, 2)


if __name__ == "__main__":
    timetest()
