import numpy as np
import numba as nb
from numba import jit

SHORT_SIZE = 16
S_PERMUTATIONS = np.array([5, 8, 0, 2, 6, 11, 1, 4, 15, 12, 3, 9, 10, 7, 13, 14])


@jit(nb.uint16(nb.uint16, nb.uint16), nopython=True)
def rotl(n, d):
    return (n << d) | (n >> (SHORT_SIZE - d))


@jit(nb.types.UniTuple(nb.uint16, 4)(nb.uint16, nb.uint16, nb.uint16, nb.uint16, nb.uint16, nb.uint16), nopython=True)
def G_function(a, b, c, d, x, y):
    a = a + b + x
    d = rotl((d ^ a), 3)
    c = c + d
    b = rotl((b ^ c), 11)
    a = a + b + y
    d = rotl((d ^ a), 2)
    c = c + d
    b = rotl((b ^ c), 5)
    return a, b, c, d


@jit(nb.uint16[:](nb.uint16[:]), nopython=True)
def permute_m_by_s(m):
    return np.array([m[S_PERMUTATIONS[i]] for i in range(SHORT_SIZE)])
