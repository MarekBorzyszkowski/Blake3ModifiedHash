import itertools

import numpy as np
import numba as nb
from src.HashFunc import blake3_hash
from numba import jit, prange

allowed_letters = 'qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM1234567890!@#$%^&*-_=+([{<)]}>\'";:?,.\\/|'
allowed_val_to_letters = {np.uint16(ord(character)): character for character in allowed_letters}
allowed_val = np.array([np.uint16(ord(character)) for character in allowed_letters])


@jit(nb.uint16(nb.uint16[:], nb.uint16[:]), nopython=True)
def compare_hash(expected_hash, actual_hash):
    for i in range(len(expected_hash)):
        if expected_hash[i] != actual_hash[i]:
            return 0
    return 1


@jit(nb.types.UniTuple(nb.uint16, 2)(nb.uint16, nb.uint16[:]), nopython=True, parallel=True)
def crack_hash(entry_message_length, expected_hash):
    number_of_elements = len(allowed_val)
    combinations = []
    result = [np.uint16(0) for _ in range(entry_message_length)]
    for a in allowed_val:
        for b in allowed_val:
            combinations.append(np.array([a, b], dtype=np.uint16))
    for i in prange(len(combinations)):
        equal = compare_hash(expected_hash, blake3_hash(combinations[i]))
        for j in range(entry_message_length):
            result[j] += equal * combinations[i][j]
    return result[0], result[1]


def convert_cracked_hash_to_string(initial_combination):
    return ''.join([allowed_val_to_letters[element] for element in initial_combination])


original_combination = crack_hash(2,
    np.array([np.uint16(0x290D), np.uint16(0x8E30), np.uint16(0xA7F7), np.uint16(0x58DE),
              np.uint16(0x023C), np.uint16(0x9C74), np.uint16(0x6233),
              np.uint16(0x631D)], dtype=np.uint16), )
print(original_combination)
original_string = convert_cracked_hash_to_string(original_combination)
print(original_string)
