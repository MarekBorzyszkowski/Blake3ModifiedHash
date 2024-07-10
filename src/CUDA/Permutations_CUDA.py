import operator

import numpy as np
import numba as nb
from numba import jit, cuda

SHORT_SIZE = 16
S_PERMUTATIONS = np.array([5, 8, 0, 2, 6, 11, 1, 4, 15, 12, 3, 9, 10, 7, 13, 14])


@cuda.jit(nb.uint32(nb.uint32, nb.uint32), device=True)
def rotl(n, d):
    return ((n << d) | (n >> SHORT_SIZE - d)) & 0xFFFF


@cuda.jit(nb.void(nb.uint32, nb.uint32, nb.uint32))
def rotl_test(n, d, output):
    output[0] = rotl(n[0], d[0])


@cuda.jit(nb.types.UniTuple(nb.uint32, 4)(nb.uint32, nb.uint32, nb.uint32, nb.uint32, nb.uint32, nb.uint32),
          device=True)
def G_function(a, b, c, d, x, y):
    a = (a + b + x) & 0xFFFF
    d = rotl((d ^ a), 3)
    c = (c + d) & 0xFFFF
    b = rotl((b ^ c), 11)
    a = (a + b + y) & 0xFFFF
    d = rotl((d ^ a), 2)
    c = (c + d) & 0xFFFF
    b = rotl((b ^ c), 5)
    return a, b, c, d


@cuda.jit(nb.void(nb.uint32[:]))
def G_function_test(array):
    array[0], array[1], array[2], array[3] = G_function(array[0], array[1], array[2], array[3], array[4], array[5])


@cuda.jit(nb.uint32[:](nb.uint32[:]), device=True)
def permute_m_by_s(m):
    return np.array([m[S_PERMUTATIONS[i]] for i in range(SHORT_SIZE)])


@cuda.jit(nb.void(nb.uint32[:]))
def permute_m_by_s_test(m):
    results = permute_m_by_s(m)
    for i in range(len(m)):
        m[i] = results[i]
