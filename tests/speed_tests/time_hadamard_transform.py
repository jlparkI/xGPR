"""A quick and dirty timeit script for checking the time to perform various
FHT operations and compare with cupy / numpy."""
import timeit
import numpy as np
import cupy as cp
import cupyx
from scipy.fftpack import dct
from xGPR.xgpr_cpu_rfgen_cpp_ext import cpuFastHadamardTransform as cFHT
from xGPR.xgpr_cpu_rfgen_cpp_ext import cpuSORFTransform as cSORF


#Running the CPU numpy test with too many columns is extremely
#expensive -- avoid running that test by accident.
MAX_COLS_MATMUL = 124#2048


def timetest():
    """Run a series of timeit tests and print the results."""
    #Change these lines depending on what dims you want to test.
    nthreads = 2
    nrows, ncols, nfeats = 2000, 1024, 2048
    if nfeats % ncols != 0:
        raise ValueError("nfeats must be an integer multiple of ncols.")

    nblocks = int(nfeats / ncols)
    ntests = 100
    print("**************************************")
    print(f"Running test with {nrows} rows, {ncols} cols, {nfeats} features")
    print("**************************************")
    matmul_setup = f"""
import numpy as np
random_seed = 123
rng = np.random.default_rng(random_seed)
marr = rng.uniform(low=-10.0, high=10.0, size=({nrows},{ncols}))
Q = rng.uniform(low=-1.0, high=1.0, size=({ncols},{nfeats}))
from __main__ import matmul_test"""
    if ncols < MAX_COLS_MATMUL:
        print("Time(us) for simple matrix multiplication:")
        time_taken = timeit.timeit("matmul_test(marr, Q)", setup=matmul_setup,
                number=ntests)
        print(1e6 * time_taken / ntests)


    fh3d_setup = f"""
nthreads = {nthreads}
import math
import numpy as np
from scipy.linalg import hadamard
from cpu_rf_gen_module import cpuFastHadamardTransform as cFHT
from cpu_rf_gen_module import cpuSORFTransform as cSORF
random_seed = 123
rng = np.random.default_rng(random_seed)
marr = rng.uniform(low=-10.0, high=10.0, size=({nrows},{nblocks},{ncols}))
radem_array = np.asarray([-1.0, 1.0])
D1 = rng.choice(radem_array, size=(3,{nblocks},{ncols}), replace=True)
from __main__ import fh3d_test"""
    print(f"Time(us) for fh3d version with {nthreads} threads:")
    time_taken = timeit.timeit("fh3d_test(marr, D1, nthreads)", setup=fh3d_setup,
                number=ntests)
    print(1e6 * time_taken / ntests)


    block_setup = f"""
nthreads = {nthreads}
import numpy as np
from scipy.linalg import hadamard
from cpu_rf_gen_module import cpuFastHadamardTransform as cFHT
from cpu_rf_gen_module import cpuSORFTransform as cSORF
random_seed = 123
rng = np.random.default_rng(random_seed)
marr = rng.uniform(low=-10.0, high=10.0, size=({nrows},{nblocks},{ncols}))
radem_array = np.asarray([-1, 1])
D1 = rng.choice(radem_array, size=(3,{nblocks},{ncols}), replace=True).astype(np.int8)
from __main__ import sorf_test"""
    print(f"Time(us) for block version with {nthreads} threads::")
    time_taken = timeit.timeit("sorf_test(marr, D1, nthreads)", setup=block_setup,
                number=ntests)
    print(1e6 * time_taken / ntests)


    '''random_seed = 123
    rng = np.random.default_rng(random_seed)
    marr = rng.uniform(low=-10.0, high=10.0, size=(nrows,nblocks,ncols))
    radem_array = np.asarray([-1, 1])
    diag1 = rng.choice(radem_array, size=(3,nblocks,ncols),
                    replace=True).astype(np.int8)
    marr = cp.asarray(marr).astype(cp.float32)
    diag1 = cp.asarray(diag1)
    print("Time for cuda version:")
    time_taken = cupyx.time.repeat(cuda_test, (marr, diag1), n_repeat = ntests)
    print(time_taken)'''


    dct_setup = f"""
import numpy as np, cupy as cp
random_seed = 123
rng = np.random.default_rng(random_seed)
marr = rng.uniform(low=-10.0, high=10.0, size=({nrows},{nblocks},{ncols}))
radem_array = np.asarray([-1, 1])
D1 = rng.choice(radem_array, size=(3,{nblocks},{ncols}), replace=True).astype(np.int8)
from __main__ import dct_test
from scipy.fftpack import dct"""
    print("Time(us) for DCT version:")
    time_taken = timeit.timeit("dct_test(marr, D1)", setup=dct_setup,
                number=ntests)
    print(1e6 * time_taken / ntests)

    random_seed = 123
    rng = np.random.default_rng(random_seed)
    marr = rng.uniform(low=-10.0, high=10.0, size=(nrows,ncols))
    qmat = rng.uniform(low=-1.0, high=1.0, size=(ncols,nfeats))
    qmat = cp.asarray(qmat).astype(cp.float32)
    marr = cp.asarray(marr).astype(cp.float32)
    print("Time for matrix multiplication with Cupy:")
    time_taken = cupyx.time.repeat(cp_matmul_test, (marr, qmat), n_repeat = ntests)
    print(time_taken)





def fh3d_test(marr, diag, nthreads):
    """Generate SORF features using the FHT module with
    separate diag matmul / fht operations."""
    marr = marr * diag[0:1,:,:]
    cFHT(marr, nthreads)
    marr = marr * diag[1:2,:,:]
    cFHT(marr, nthreads)
    marr = marr * diag[2:3,:,:]
    cFHT(marr, nthreads)


def sorf_test(marr, diag, nthreads):
    """Generate SORF features using the sorf function of
    the module."""
    cSORF(marr, diag, nthreads)


def matmul_test(marr, qmat):
    """Ordinary matrix multiplication."""
    _ = marr @ qmat


def cp_matmul_test(marr, qmat):
    """Matrix multiplication on CUDA."""
    _ = cp.matmul(marr, qmat)


def dct_test(marr, diag):
    """How fast would Scipy's DCT be by comparison?"""
    marr = marr * diag[0:1,:,:]
    marr = dct(marr, axis=-1, norm="ortho")
    marr = marr * diag[1:2,:,:]
    marr = dct(marr, axis=-1, norm="ortho")
    marr = marr * diag[2:3,:,:]
    marr = dct(marr, axis=-1, norm="ortho")

if __name__ == "__main__":
    timetest()
