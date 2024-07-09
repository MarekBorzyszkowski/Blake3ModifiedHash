import itertools

import numpy as np
import numba as nb
from src.HashFunc import blake3_hash
from numba import jit, prange

allowed_letters = 'qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM1234567890!@#$%^&*-_=+([{<)]}>\'";:?,.\\/|'
allowed_val_to_letters = {np.uint16(ord(character)): character for character in allowed_letters}
allowed_val = np.array([np.uint16(ord(character)) for character in allowed_letters], dtype=np.uint16)


@jit(nb.uint16[:](nb.uint64, nb.uint64), nopython=True)
def get_combination(entry_message_length, number):
    indexes = []
    for i in range(entry_message_length):
        indexes.append(np.uint16(number % len(allowed_val)))
        number = number // len(allowed_val)
    return np.array([np.uint16(allowed_val[index]) for index in indexes], dtype=np.uint16)


@jit(nb.uint16(nb.uint16[:], nb.uint16[:]), nopython=True)
def compare_hash(expected_hash, actual_hash):
    for i in range(len(expected_hash)):
        if expected_hash[i] != actual_hash[i]:
            return 0
    return 1


@jit(nb.uint16[:](nb.uint64, nb.uint16[:]), nopython=True, parallel=True)
def crack_hash(entry_message_length, expected_hash):
    number_of_elements = len(allowed_val)
    result = np.array([np.uint16(0) for _ in range(entry_message_length)], dtype=np.uint16)
    for i in prange(number_of_elements ** entry_message_length):
        combination = get_combination(entry_message_length, i)
        equal = compare_hash(expected_hash, blake3_hash(combination))
        for j in range(entry_message_length):
            result[j] += equal * combination[j]
    return result


def convert_cracked_hash_to_string(initial_combination):
    return ''.join([allowed_val_to_letters[element] for element in initial_combination])


cracking_presets = {
    2: np.array([np.uint16(0x290D), np.uint16(0x8E30), np.uint16(0xA7F7), np.uint16(0x58DE),
                 np.uint16(0x023C), np.uint16(0x9C74), np.uint16(0x6233), np.uint16(0x631D)], dtype=np.uint16),
    3: np.array([np.uint16(0x6C34), np.uint16(0x6E8D), np.uint16(0x3067), np.uint16(0xEF3B),
                 np.uint16(0x7BC3), np.uint16(0xE5C2), np.uint16(0x99CC), np.uint16(0x7535)], dtype=np.uint16),
    4: np.array([np.uint16(0xE14D), np.uint16(0xA6D5), np.uint16(0xEB17), np.uint16(0x15BE),
                 np.uint16(0xCD5D), np.uint16(0x4680), np.uint16(0xD99D), np.uint16(0x6EDC)], dtype=np.uint16),
    5: np.array([np.uint16(0x268D), np.uint16(0xDEE3), np.uint16(0xCD85), np.uint16(0x4D73),
                 np.uint16(0x80E5), np.uint16(0x4F61), np.uint16(0x5712), np.uint16(0x86CD)], dtype=np.uint16),
    6: np.array([np.uint16(0xCFAC), np.uint16(0x5548), np.uint16(0x46A0), np.uint16(0x7CF5),
                 np.uint16(0x5434), np.uint16(0x4C38), np.uint16(0x7B8E), np.uint16(0x48DC)], dtype=np.uint16),
    7: np.array([np.uint16(0x22AA), np.uint16(0x7576), np.uint16(0x758A), np.uint16(0x3978),
                 np.uint16(0x77BC), np.uint16(0x3AA0), np.uint16(0x40F5), np.uint16(0xBD12)], dtype=np.uint16),
    8: np.array([np.uint16(0x6207), np.uint16(0x2580), np.uint16(0x374C), np.uint16(0x71E6),
                 np.uint16(0x0D2C), np.uint16(0x835E), np.uint16(0x3398), np.uint16(0x5BE5)], dtype=np.uint16),
}
length = 4
original_combination = crack_hash(length, cracking_presets[length])
print(original_combination)
original_string = convert_cracked_hash_to_string(original_combination)
print(original_string)
