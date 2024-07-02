"""Wraps the C functions that perform the fastHadamardTransform on CPU,
the structured orthogonal features or SORF operations on CPU and the
fast Hadamard transform based SRHT on CPU.

Also performs all of the bounds and safety checks needed to use these
functions (the C functions do not do their own bounds checking). It
is EXTREMELY important that this wrapper not be bypassed for this
reason -- it double checks all of the array dimensions, types,
is data contiguous etc. before calling the wrapped C functions."""
import numpy as np
cimport numpy as np
cimport cython
from cython cimport floating
from libc cimport stdint
from libc.stdint cimport uintptr_t
from libc.stdint cimport int8_t
import math


