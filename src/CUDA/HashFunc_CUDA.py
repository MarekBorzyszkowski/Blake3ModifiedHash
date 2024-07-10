import numba as nb
import numpy as np
from numba import cuda

from Permutations_CUDA import G_function, permute_m_by_s


@cuda.jit(device=True)
def vertical_permutation(v, m):
    G_function(v[0], v[4], v[8], v[12], m[0], m[1])
    G_function(v[1], v[5], v[9], v[13], m[2], m[3])
    G_function(v[2], v[6], v[10], v[14], m[4], m[5])
    G_function(v[3], v[7], v[11], v[15], m[6], m[7])


@cuda.jit(device=True)
def diagonal_permutation(v, m):
    G_function(v[0], v[5], v[10], v[15], m[8], m[9])
    G_function(v[1], v[6], v[11], v[12], m[10], m[11])
    G_function(v[2], v[7], v[8], v[13], m[12], m[13])
    G_function(v[3], v[4], v[9], v[14], m[14], m[15])


@cuda.jit(device=True)
def make_round(v, m):
    vertical_permutation(v, m)
    diagonal_permutation(v, m)
    permute_m_by_s(m)


@cuda.jit(device=True)
def hash_block(w, m, block_number):
    v = np.array([w[0], w[1], w[2], w[3],
                  w[4], w[5], w[6], w[7],
                  np.uint32(0x03F4), np.uint32(0x774C), np.uint32(0x5690), np.uint32(0xC878),
                  np.uint32(0), np.uint32(block_number), np.uint32(0), np.uint32(0)])
    for i in range(6):
        make_round(v, m)
    return np.array([w[0] ^ v[0] ^ v[8], w[1] ^ v[1] ^ v[9], w[2] ^ v[2] ^ v[10], w[3] ^ v[3] ^ v[11],
                     w[4] ^ v[4] ^ v[12], w[5] ^ v[5] ^ v[13], w[6] ^ v[6] ^ v[14], w[7] ^ v[7] ^ v[15]],
                    dtype=np.uint32)


@cuda.jit(device=True)
def merge_bytes(block_as_bytes):
    return np.array([np.uint32((block_as_bytes[2 * i] << 8) + block_as_bytes[2 * i + 1])
                     for i in range(len(block_as_bytes) // 2)])


@cuda.jit(device=True)
def fill_blocks(block):
    number_of_elements = block.shape[0]
    number_of_missing_elements = 32 - number_of_elements % 32
    if number_of_missing_elements == 32:
        block = np.append(block, np.array(
            [np.uint32(0x007F), np.uint32(0x00FF), np.uint32(0x00FF), np.uint32(0x00FF),
             np.uint32(0x00FF), np.uint32(0x00FF), np.uint32(0x00FF), np.uint32(0x00FF),
             np.uint32(0x00FF), np.uint32(0x00FF), np.uint32(0x00FF), np.uint32(0x00FF),
             np.uint32(0x00FF), np.uint32(0x00FF), np.uint32(0x00FF), np.uint32(0x00FF),
             np.uint32(0x00FF), np.uint32(0x00FF), np.uint32(0x00FF), np.uint32(0x00FF),
             np.uint32(0x00FF), np.uint32(0x00FF), np.uint32(0x00FF), np.uint32(0x00FF),
             np.uint32(0x00FF), np.uint32(0x00FF), np.uint32(0x00FF), np.uint32(0x00FF),
             np.uint32(0x00FF), np.uint32(0x00FF), np.uint32(0x00FF), np.uint32(0x00FF)]))
    else:
        block = np.append(block, np.array(np.uint32(0x007F)))
        number_of_missing_elements -= 1
        if number_of_missing_elements > 0:
            block = np.append(block, np.array([np.uint32(0x00FF) for _ in range(number_of_missing_elements)]))
    return block


@cuda.jit(device=True)
def blake3_hash(block_of_bytes):
    # block_of_bytes = fill_blocks(block_of_bytes)
    block_of_words = block_of_bytes#merge_bytes(block_of_bytes)
    w = np.array([np.uint32(0) for _ in range(8)])
    number_of_blocks = len(block_of_words) // 16
    for i in range(number_of_blocks):
        w = hash_block(w, block_of_words[16 * i:16 * i + 16], i)
    return w


allowed_letters = 'qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM1234567890!@#$%^&*-_=+([{<)]}>\'";:?,.\\/|'
allowed_val_to_letters = {np.uint32(ord(character)): character for character in allowed_letters}
allowed_val = np.array([np.uint32(ord(character)) for character in allowed_letters], dtype=np.uint32)

@cuda.jit(device=True)
def get_combination(entry_message_length, number, combination):
    for i in range(entry_message_length):
        combination[i] = (np.uint32(allowed_val[number % len(allowed_val)]))
        number = number // len(allowed_val)


@cuda.jit(device=True)
def compare_hash(expected_hash, actual_hash):
    for i in range(len(expected_hash)):
        if expected_hash[i] != actual_hash[i]:
            return 0
    return 1
