import unittest

import numpy as np
from numba import cuda

from src.CUDA.Permutations_CUDA import rotl_test

THREADS_PER_BLOCK = 128
BLOCKS_PER_GRID = 16


class MyTestCase(unittest.TestCase):
    def test_rol_one_one_shit(self):
        number = np.array([1], dtype=np.int32)
        shift = np.array([1], dtype=np.int32)
        cuda_result = cuda.device_array([1], dtype=np.int32)
        expected = 2
        rotl_test[1, 1](number, shift, cuda_result)
        result = cuda_result.copy_to_host()
        self.assertEqual(expected, result[0])  # add assertion here

    def test_rol_shift_with_one_going_to_beginning(self):
        number = np.uint16(32769)
        shift = 1
        expected = np.uint16(3)
        self.assertEqual(expected, rotl_test(number, shift))  # add assertion here

    def test_rol_number_length_rol(self):
        number = np.uint16(1234)
        shift = 16
        expected = np.uint16(1234)
        self.assertEqual(expected, rotl_test(number, shift))  # add assertion here


if __name__ == '__main__':
    unittest.main()
