import re

import numpy as np
import numba as nb
from numba import jit
from src.Permutations import G_function, permute_m_by_s


#@jit(nopython=True)
def blake3_hash(message):
    if len(message) is None:
        message = ''
    blocks = re.findall('.{1,32}', message)
    blocks_as_bytes = [np.array([np.uint16(ord(character)) for character in block]) for block in blocks]
    if len(blocks_as_bytes) == 0:
        blocks_as_bytes.append(np.empty(0, dtype=np.uint16))
    blocks_as_bytes = fill_blocks(blocks_as_bytes)
    blocks_as_words = merge_bytes(blocks_as_bytes)
    w = np.array([np.uint16(0) for _ in range(8)])
    for i in range(len(blocks_as_words)):
        w = hash_block(w, blocks_as_words[i], i)
    return w


#@jit(nopython=True)
def merge_bytes(blocks_as_bytes):
    return [np.array([np.uint16((block[2*i] << 8) + block[2*i+1]) for i in range(16)]) for block in blocks_as_bytes]


#@jit(nopython=True)
def fill_blocks(blocks):
    if len(blocks[-1]) == 32:
        blocks.append(np.array(
            [np.uint16(0x007F), np.uint16(0x00FF), np.uint16(0x00FF), np.uint16(0x00FF),
             np.uint16(0x00FF), np.uint16(0x00FF), np.uint16(0x00FF), np.uint16(0x00FF),
             np.uint16(0x00FF), np.uint16(0x00FF), np.uint16(0x00FF), np.uint16(0x00FF),
             np.uint16(0x00FF), np.uint16(0x00FF), np.uint16(0x00FF), np.uint16(0x00FF),
             np.uint16(0x00FF), np.uint16(0x00FF), np.uint16(0x00FF), np.uint16(0x00FF),
             np.uint16(0x00FF), np.uint16(0x00FF), np.uint16(0x00FF), np.uint16(0x00FF),
             np.uint16(0x00FF), np.uint16(0x00FF), np.uint16(0x00FF), np.uint16(0x00FF),
             np.uint16(0x00FF), np.uint16(0x00FF), np.uint16(0x00FF), np.uint16(0x00FF)]))
    else:
        blocks[-1] = np.append(blocks[-1], np.uint16(0x007F))
        num_of_elements_in_last_block = len(blocks[-1])
        for i in range(num_of_elements_in_last_block, 32):
            blocks[-1] = np.append(blocks[-1], np.uint16(0x00FF))
    return blocks


#@jit(nb.uint16[:](nb.uint16[:], nb.uint16[:], nb.uint16), nopython=True)
def hash_block(w, m, block_number):
    v = np.array([w[0], w[1], w[2], w[3],
                  w[4], w[5], w[6], w[7],
                  np.uint16(0x03F4), np.uint16(0x774C), np.uint16(0x5690), np.uint16(0xC878),
                  np.uint16(0), np.uint16(block_number), np.uint16(0), np.uint16(0)])
    for i in range(6):
        v, m = make_round(v, m)
    return np.array([w[0] ^ v[0] ^ v[8], w[1] ^ v[1] ^ v[9], w[2] ^ v[2] ^ v[10], w[3] ^ v[3] ^ v[11],
                     w[4] ^ v[4] ^ v[12], w[5] ^ v[5] ^ v[13], w[6] ^ v[6] ^ v[14], w[7] ^ v[7] ^ v[15]])


#@jit(nb.types.UniTuple(nb.uint16[:], 2)(nb.uint16[:], nb.uint16[:]), nopython=True)
def make_round(v, m):
    v = vertical_permutation(v, m)
    v = diagonal_permutation(v, m)
    m = permute_m_by_s(m)
    return v, m


#@jit(nb.uint16[:](nb.uint16[:], nb.uint16[:]), nopython=True)
def vertical_permutation(v, m):
    v[0], v[4], v[8], v[12] = G_function(v[0], v[4], v[8], v[12], m[0], m[1])
    v[1], v[5], v[9], v[13] = G_function(v[1], v[5], v[9], v[13], m[2], m[3])
    v[2], v[6], v[10], v[14] = G_function(v[2], v[6], v[10], v[14], m[4], m[5])
    v[3], v[7], v[11], v[15] = G_function(v[3], v[7], v[11], v[15], m[6], m[7])
    return v


#@jit(nb.uint16[:](nb.uint16[:], nb.uint16[:]), nopython=True)
def diagonal_permutation(v, m):
    v[0], v[5], v[10], v[15] = G_function(v[0], v[5], v[10], v[15], m[8], m[9])
    v[1], v[6], v[11], v[12] = G_function(v[1], v[6], v[11], v[12], m[10], m[11])
    v[2], v[7], v[8], v[13] = G_function(v[2], v[7], v[8], v[13], m[12], m[13])
    v[3], v[4], v[9], v[14] = G_function(v[3], v[4], v[9], v[14], m[14], m[15])
    return v
