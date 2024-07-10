import unittest

import numpy as np
from numba import cuda

from src.CUDA.HashFunc_CUDA import vertical_permutation_test, diagonal_permutation_test, hash_block_test, \
    message_to_binary, blake3_hash_test


class MyTestCase(unittest.TestCase):
    def test_vertical_permutation(self):
        v = np.array([np.uint32(0), np.uint32(0), np.uint32(0), np.uint32(0),
                      np.uint32(0), np.uint32(0), np.uint32(0), np.uint32(0),
                      np.uint32(0x03F4), np.uint32(0x774C), np.uint32(0x5690), np.uint32(0xC878),
                      np.uint32(0), np.uint32(0), np.uint32(0), np.uint32(0)])
        m = np.array([np.uint32(((2 * i) << 8) + (2 * i + 1)) for i in range(16)])
        expected = np.array([np.uint32(0xE223), np.uint32(0xAEC7), np.uint32(0xD6CA), np.uint32(0x1B63),
                             np.uint32(0x968D), np.uint32(0xF12E), np.uint32(0x8A86), np.uint32(0x8942),
                             np.uint32(0x8CAB), np.uint32(0xD332), np.uint32(0xF0E2), np.uint32(0x150D),
                             np.uint32(0x88AF), np.uint32(0x3BBE), np.uint32(0x5A0A), np.uint32(0xEC2D)])
        cuda_v = cuda.to_device(v)
        cuda_m = cuda.to_device(m)
        vertical_permutation_test[1, 1](cuda_v, cuda_m)
        results = cuda_v.copy_to_host()
        for i in range(len(v)):
            self.assertEqual(expected[i], results[i])  # add assertion here

    def test_diagonal_permutation(self):
        v = np.array([np.uint32(0xE223), np.uint32(0xAEC7), np.uint32(0xD6CA), np.uint32(0x1B63),
                      np.uint32(0x968D), np.uint32(0xF12E), np.uint32(0x8A86), np.uint32(0x8942),
                      np.uint32(0x8CAB), np.uint32(0xD332), np.uint32(0xF0E2), np.uint32(0x150D),
                      np.uint32(0x88AF), np.uint32(0x3BBE), np.uint32(0x5A0A), np.uint32(0xEC2D)])
        m = np.array([np.uint32(((2 * i) << 8) + (2 * i + 1)) for i in range(16)])
        expected = np.array([np.uint32(0x9A48), np.uint32(0x51C8), np.uint32(0xCB46), np.uint32(0x0B5B),
                             np.uint32(0xC467), np.uint32(0x19C9), np.uint32(0x8B75), np.uint32(0xDFC7),
                             np.uint32(0x07F8), np.uint32(0x210C), np.uint32(0xEC1D), np.uint32(0x4214),
                             np.uint32(0xFE99), np.uint32(0x5E73), np.uint32(0xAD9E), np.uint32(0x80C3)])
        cuda_v = cuda.to_device(v)
        cuda_m = cuda.to_device(m)
        diagonal_permutation_test[1, 1](cuda_v, cuda_m)
        results = cuda_v.copy_to_host()
        for i in range(len(v)):
            self.assertEqual(expected[i], results[i])  # add assertion here

    def test_hash_block(self):
        m = np.array([np.uint32(((2 * i) << 8) + (2 * i + 1)) for i in range(16)])
        init_w = np.array([np.uint32(0) for _ in range(8)])
        expected = np.array([np.uint32(0xF089), np.uint32(0x4377), np.uint32(0x32AC), np.uint32(0x4197),
                             np.uint32(0x63C3), np.uint32(0x975A), np.uint32(0x15CD), np.uint32(0xDD5B)])
        cuda_m = cuda.to_device(m)
        cuda_w = cuda.to_device(init_w)
        hash_block_test[1, 1](cuda_w, cuda_m, 0)
        result = cuda_w.copy_to_host()
        for i in range(len(result)):
            self.assertEqual(expected[i], result[i])  # add assertion here

    def test_blake3_hash_1(self):
        message = ''
        expected = np.array([np.uint32(0x898F), np.uint32(0xE038), np.uint32(0xCC44), np.uint32(0xAC95),
                             np.uint32(0x0F78), np.uint32(0xF84D), np.uint32(0x8796), np.uint32(0x98C9)])
        binary = message_to_binary(message)
        cuda_message = cuda.to_device(binary)
        cuda_w = cuda.device_array([8], np.uint32)
        blake3_hash_test[1, 1](cuda_message, cuda_w)
        result = cuda_w.copy_to_host()
        for i in range(len(expected)):
            self.assertEqual(expected[i], result[i])

    def test_blake3_hash_2(self):
        message = 'AbCxYz'
        expected = np.array([np.uint32(0xE1C1), np.uint32(0x3F52), np.uint32(0x3C78), np.uint32(0x7589),
                             np.uint32(0x22FD), np.uint32(0x11AA), np.uint32(0x3132), np.uint32(0xD01C)])
        binary = message_to_binary(message)
        cuda_message = cuda.to_device(binary)
        cuda_w = cuda.device_array([8], np.uint32)
        blake3_hash_test[1, 1](cuda_message, cuda_w)
        result = cuda_w.copy_to_host()
        for i in range(len(expected)):
            self.assertEqual(expected[i], result[i])

    def test_blake3_hash_3(self):
        message = '1234567890'
        expected = np.array([np.uint32(0x8691), np.uint32(0x1F68), np.uint32(0xBF45), np.uint32(0xA5D6),
                             np.uint32(0xC295), np.uint32(0xB6F7), np.uint32(0x95D9), np.uint32(0xB9BE)])
        binary = message_to_binary(message)
        cuda_message = cuda.to_device(binary)
        cuda_w = cuda.device_array([8], np.uint32)
        blake3_hash_test[1, 1](cuda_message, cuda_w)
        result = cuda_w.copy_to_host()
        for i in range(len(expected)):
            self.assertEqual(expected[i], result[i])

    def test_blake3_hash_4(self):
        message = 'Ala ma kota, kot ma ale.'
        expected = np.array([np.uint32(0xB0E3), np.uint32(0x5AD8), np.uint32(0xBCC3), np.uint32(0x0D12),
                             np.uint32(0x2FED), np.uint32(0xA609), np.uint32(0xDE3C), np.uint32(0x991C)])
        binary = message_to_binary(message)
        cuda_message = cuda.to_device(binary)
        cuda_w = cuda.device_array([8], np.uint32)
        blake3_hash_test[1, 1](cuda_message, cuda_w)
        result = cuda_w.copy_to_host()
        for i in range(len(expected)):
            self.assertEqual(expected[i], result[i])

    def test_blake3_hash_5(self):
        message = 'Ty, ktory wchodzisz, zegnaj sie z nadzieja.'
        expected = np.array([np.uint32(0x862B), np.uint32(0xEA4A), np.uint32(0x8377), np.uint32(0xCB1C),
                             np.uint32(0x7CF2), np.uint32(0x1851), np.uint32(0xF729), np.uint32(0xD593)])
        binary = message_to_binary(message)
        cuda_message = cuda.to_device(binary)
        cuda_w = cuda.device_array([8], np.uint32)
        blake3_hash_test[1, 1](cuda_message, cuda_w)
        result = cuda_w.copy_to_host()
        for i in range(len(expected)):
            self.assertEqual(expected[i], result[i])

    def test_blake3_hash_6(self):
        message = 'Litwo, Ojczyzno moja! ty jestes jak zdrowie;'
        #
        expected = np.array([np.uint32(0x94FE), np.uint32(0x5359), np.uint32(0x63CD), np.uint32(0x4055),
                             np.uint32(0xAA16), np.uint32(0x2206), np.uint32(0x5A34), np.uint32(0x55A5)])
        binary = message_to_binary(message)
        cuda_message = cuda.to_device(binary)
        cuda_w = cuda.device_array([8], np.uint32)
        blake3_hash_test[1, 1](cuda_message, cuda_w)
        result = cuda_w.copy_to_host()
        for i in range(len(expected)):
            self.assertEqual(expected[i], result[i])

    def test_blake3_hash_7(self):
        message = 'a'*48000
        expected = np.array([np.uint32(0x738C), np.uint32(0x652D), np.uint32(0x7274), np.uint32(0xEFC3),
                             np.uint32(0xB8F4), np.uint32(0x804C), np.uint32(0xDC2D), np.uint32(0x2873)])
        binary = message_to_binary(message)
        cuda_message = cuda.to_device(binary)
        cuda_w = cuda.device_array([8], np.uint32)
        blake3_hash_test[1, 1](cuda_message, cuda_w)
        result = cuda_w.copy_to_host()
        for i in range(len(expected)):
            self.assertEqual(expected[i], result[i])

    def test_blake3_hash_8(self):
        message = 'a'*48479
        expected = np.array([np.uint32(0x3705), np.uint32(0xB383), np.uint32(0xC5F6), np.uint32(0x199B),
                             np.uint32(0x874D), np.uint32(0xD66A), np.uint32(0x8BB0), np.uint32(0xE749)])
        binary = message_to_binary(message)
        cuda_message = cuda.to_device(binary)
        cuda_w = cuda.device_array([8], np.uint32)
        blake3_hash_test[1, 1](cuda_message, cuda_w)
        result = cuda_w.copy_to_host()
        for i in range(len(expected)):
            self.assertEqual(expected[i], result[i])

    def test_blake3_hash_9(self):
        message = 'a'*48958
        expected = np.array([np.uint32(0xDB87), np.uint32(0xB2C0), np.uint32(0xC169), np.uint32(0xA785),
                             np.uint32(0x96E3), np.uint32(0x2814), np.uint32(0x5B46), np.uint32(0xBFAC)])
        binary = message_to_binary(message)
        cuda_message = cuda.to_device(binary)
        cuda_w = cuda.device_array([8], np.uint32)
        blake3_hash_test[1, 1](cuda_message, cuda_w)
        result = cuda_w.copy_to_host()
        for i in range(len(expected)):
            self.assertEqual(expected[i], result[i])


if __name__ == '__main__':
    unittest.main()
