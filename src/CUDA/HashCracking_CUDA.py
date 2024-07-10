import time

import numba as nb
import numpy as np
from numba import cuda

from HashFunc_CUDA import blake3_hash

allowed_letters = 'qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM1234567890!@#$%^&*-_=+([{<)]}>\'";:?,.\\/|'
allowed_val_to_letters = {np.uint32(ord(character)): character for character in allowed_letters}
allowed_val = np.array([np.uint32(ord(character)) for character in allowed_letters], dtype=np.uint32)

THREADS_PER_BLOCK = 128
BLOCKS_PER_GRID = 16

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


@cuda.jit
def crack_hash(entry_message_length, expected_hash, result):
    combination = cuda.local.array(2, nb.types.uint32)
    # tid = cuda.threadIdx.x
    # bid = cuda.blockIdx.x
    # bdim = cuda.blockDim.x
    # gdim = cuda.gridDim.x
    # beginning = (bid * bdim) + tid
    # number_of_threads = bdim * gdim
    # number_of_elements = len(allowed_val)
    # for i in range(beginning, number_of_elements ** entry_message_length, number_of_threads):
    #     get_combination(entry_message_length, i, combination)
    #     equal = compare_hash(expected_hash, blake3_hash(combination))
    #     for j in range(entry_message_length):
    #         if equal == 1:
    #             cuda.atomic.add(result, j, combination[j])


def convert_cracked_hash_to_string(initial_combination):
    return ''.join([allowed_val_to_letters[element] for element in initial_combination])


cracking_presets = {
    2: np.array([np.uint32(0x290D), np.uint32(0x8E30), np.uint32(0xA7F7), np.uint32(0x58DE),
                 np.uint32(0x023C), np.uint32(0x9C74), np.uint32(0x6233), np.uint32(0x631D)], dtype=np.uint32),
    3: np.array([np.uint32(0x6C34), np.uint32(0x6E8D), np.uint32(0x3067), np.uint32(0xEF3B),
                 np.uint32(0x7BC3), np.uint32(0xE5C2), np.uint32(0x99CC), np.uint32(0x7535)], dtype=np.uint32),
    4: np.array([np.uint32(0xE14D), np.uint32(0xA6D5), np.uint32(0xEB17), np.uint32(0x15BE),
                 np.uint32(0xCD5D), np.uint32(0x4680), np.uint32(0xD99D), np.uint32(0x6EDC)], dtype=np.uint32),
    5: np.array([np.uint32(0x268D), np.uint32(0xDEE3), np.uint32(0xCD85), np.uint32(0x4D73),
                 np.uint32(0x80E5), np.uint32(0x4F61), np.uint32(0x5712), np.uint32(0x86CD)], dtype=np.uint32),
    6: np.array([np.uint32(0xCFAC), np.uint32(0x5548), np.uint32(0x46A0), np.uint32(0x7CF5),
                 np.uint32(0x5434), np.uint32(0x4C38), np.uint32(0x7B8E), np.uint32(0x48DC)], dtype=np.uint32),
    7: np.array([np.uint32(0x22AA), np.uint32(0x7576), np.uint32(0x758A), np.uint32(0x3978),
                 np.uint32(0x77BC), np.uint32(0x3AA0), np.uint32(0x40F5), np.uint32(0xBD12)], dtype=np.uint32),
    8: np.array([np.uint32(0x6207), np.uint32(0x2580), np.uint32(0x374C), np.uint32(0x71E6),
                 np.uint32(0x0D2C), np.uint32(0x835E), np.uint32(0x3398), np.uint32(0x5BE5)], dtype=np.uint32),
}


length = 2
print(f"Start cracking for length {length}")
cuda_results = cuda.to_device(np.array([0 for _ in range(length)], dtype=np.uint32))
start = time.perf_counter()
crack_hash[1, 1](length, cracking_presets[length], cuda_results)
finish = time.perf_counter()
results = cuda_results.copy_to_host()
print(f"Elapsed time: {finish - start} s")
print(results)
original_string = convert_cracked_hash_to_string(results)
print(original_string)
