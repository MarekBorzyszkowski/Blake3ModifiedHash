import unittest

import numpy as np
import numba as nb
from numba import cuda

from src.CUDA.Permutations_CUDA import rotl, G_function, permute_m_by_s


@cuda.jit(nb.void(nb.uint32, nb.uint32, nb.uint32))
def rotl_test(n, d, output):
    output[0] = rotl(n[0], d[0])


@cuda.jit(nb.void(nb.uint32[:]))
def G_function_test(array):
    array[0], array[1], array[2], array[3] = G_function(array[0], array[1], array[2], array[3], array[4], array[5])


@cuda.jit(nb.void(nb.uint32[:]))
def permute_m_by_s_test(m):
    permute_m_by_s(m)


class MyTestCase(unittest.TestCase):
    def test_rol_one_one_shit(self):
        number = np.array([1], dtype=np.int32)
        shift = np.array([1], dtype=np.int32)
        cuda_result = cuda.device_array([1], dtype=np.int32)
        expected = 2
        rotl_test[1, 1](number, shift, cuda_result)
        result = cuda_result.copy_to_host()
        self.assertEqual(expected, result[0])

    def test_rol_shift_with_one_going_to_beginning(self):
        number = np.array([32769], dtype=np.int32)
        shift = np.array([1], dtype=np.int32)
        cuda_result = cuda.device_array([1], dtype=np.int32)
        expected = 3
        rotl_test[1, 1](number, shift, cuda_result)
        result = cuda_result.copy_to_host()
        self.assertEqual(expected, result[0])

    def test_rol_number_length_rol(self):
        number = np.array([1234], dtype=np.int32)
        shift = np.array([16], dtype=np.int32)
        cuda_result = cuda.device_array([1], dtype=np.int32)
        expected = 1234
        rotl_test[1, 1](number, shift, cuda_result)
        result = cuda_result.copy_to_host()
        self.assertEqual(expected, result[0])

    def test_G_function(self):
        v00 = 0
        v10 = 0
        v20 = 0x03F4
        v30 = 0
        m0 = 0x0001
        m1 = 0x0203
        cuda_array = cuda.to_device(np.array([v00, v10, v20, v30, m0, m1], dtype=np.int32))
        G_function_test[1, 1](cuda_array)
        result = cuda_array.copy_to_host()
        self.assertEqual(0xE223, result[0])
        self.assertEqual(0x968D, result[1])
        self.assertEqual(0x8CAB, result[2])
        self.assertEqual(0x88AF, result[3])

    def test_permute_m_by_s(self):
        m = np.array([np.uint16(((2 * i) << 8) + (2 * i + 1)) for i in range(16)])
        expected = np.array([np.uint16(0x0A0B), np.uint16(0x1011), np.uint16(0x0001), np.uint16(0x0405),
                             np.uint16(0x0C0D), np.uint16(0x1617), np.uint16(0x0203), np.uint16(0x0809),
                             np.uint16(0x1E1F), np.uint16(0x1819), np.uint16(0x0607), np.uint16(0x1213),
                             np.uint16(0x1415), np.uint16(0x0E0F), np.uint16(0x1A1B), np.uint16(0x1C1D)])
        cuda_array = cuda.to_device(m)
        permute_m_by_s_test[1, 1](cuda_array)
        result = cuda_array.copy_to_host()
        for i in range(len(m)):
            self.assertEqual(expected[i], result[i])


if __name__ == '__main__':
    unittest.main()
