import numpy as np
from numba import cuda

SHORT_SIZE = 16
S_PERMUTATIONS = np.array([5, 8, 0, 2, 6, 11, 1, 4, 15, 12, 3, 9, 10, 7, 13, 14])


@cuda.jit(device=True)
def rotl(n, d):
    return ((n << d) | (n >> SHORT_SIZE - d)) & 0xFFFF


@cuda.jit(device=True)
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


@cuda.jit(device=True)
def permute_m_by_s(m):
    m5 = m[5]
    i = 5
    while i != 0:
        m[i] = m[S_PERMUTATIONS[i]]
        i = S_PERMUTATIONS[i]
    m[0] = m5
    m6 = m[6]
    i = 6
    while i != 4:
        m[i] = m[S_PERMUTATIONS[i]]
        i = S_PERMUTATIONS[i]
    m[4] = m6