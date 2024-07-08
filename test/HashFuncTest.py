import unittest

import numpy as np

from src.HashFunc import vertical_permutation, diagonal_permutation, hash_block, blake3_hash


class MyTestCase(unittest.TestCase):
    def test_vertical_permutation(self):
        v = np.array([np.uint16(0), np.uint16(0), np.uint16(0), np.uint16(0),
                      np.uint16(0), np.uint16(0), np.uint16(0), np.uint16(0),
                      np.uint16(0x03F4), np.uint16(0x774C), np.uint16(0x5690), np.uint16(0xC878),
                      np.uint16(0), np.uint16(0), np.uint16(0), np.uint16(0)])
        m = np.array([np.uint16(((2 * i) << 8) + (2 * i + 1)) for i in range(16)])
        expected = np.array([np.uint16(0xE223), np.uint16(0xAEC7), np.uint16(0xD6CA), np.uint16(0x1B63),
                             np.uint16(0x968D), np.uint16(0xF12E), np.uint16(0x8A86), np.uint16(0x8942),
                             np.uint16(0x8CAB), np.uint16(0xD332), np.uint16(0xF0E2), np.uint16(0x150D),
                             np.uint16(0x88AF), np.uint16(0x3BBE), np.uint16(0x5A0A), np.uint16(0xEC2D)])
        v = vertical_permutation(v, m)
        for i in range(len(v)):
            self.assertEqual(expected[i], v[i])  # add assertion here

    def test_diagonal_permutation(self):
        v = np.array([np.uint16(0xE223), np.uint16(0xAEC7), np.uint16(0xD6CA), np.uint16(0x1B63),
                      np.uint16(0x968D), np.uint16(0xF12E), np.uint16(0x8A86), np.uint16(0x8942),
                      np.uint16(0x8CAB), np.uint16(0xD332), np.uint16(0xF0E2), np.uint16(0x150D),
                      np.uint16(0x88AF), np.uint16(0x3BBE), np.uint16(0x5A0A), np.uint16(0xEC2D)])
        m = np.array([np.uint16(((2 * i) << 8) + (2 * i + 1)) for i in range(16)])
        expected = np.array([np.uint16(0x9A48), np.uint16(0x51C8), np.uint16(0xCB46), np.uint16(0x0B5B),
                             np.uint16(0xC467), np.uint16(0x19C9), np.uint16(0x8B75), np.uint16(0xDFC7),
                             np.uint16(0x07F8), np.uint16(0x210C), np.uint16(0xEC1D), np.uint16(0x4214),
                             np.uint16(0xFE99), np.uint16(0x5E73), np.uint16(0xAD9E), np.uint16(0x80C3)])
        v = diagonal_permutation(v, m)
        for i in range(len(v)):
            self.assertEqual(expected[i], v[i])  # add assertion here

    def test_hash_block(self):
        m = np.array([np.uint16(((2 * i) << 8) + (2 * i + 1)) for i in range(16)])
        init_w = np.array([np.uint16(0) for _ in range(8)])
        expected = np.array([np.uint16(0xF089), np.uint16(0x4377), np.uint16(0x32AC), np.uint16(0x4197),
                             np.uint16(0x63C3), np.uint16(0x975A), np.uint16(0x15CD), np.uint16(0xDD5B)])
        result = hash_block(init_w, m, 0)
        for i in range(len(result)):
            self.assertEqual(expected[i], result[i])  # add assertion here

    def test_blake3_hash_empty(self):
        message = ''
        expected = np.array([np.uint16(0x898F), np.uint16(0xE038), np.uint16(0xCC44), np.uint16(0xAC95),
                             np.uint16(0x0F78), np.uint16(0xF84D), np.uint16(0x8796), np.uint16(0x98C9)])
        result = blake3_hash(message)
        for i in range(len(expected)):
            self.assertEqual(expected[i], result[i])

    def test_blake3_hash_2(self):
        message = 'AbCxYz'
        expected = np.array([np.uint16(0xE1C1), np.uint16(0x3F52), np.uint16(0x3C78), np.uint16(0x7589),
                             np.uint16(0x22FD), np.uint16(0x11AA), np.uint16(0x3132), np.uint16(0xD01C)])
        result = blake3_hash(message)
        for i in range(len(expected)):
            self.assertEqual(expected[i], result[i])

    def test_blake3_hash_3(self):
        message = '1234567890'
        expected = np.array([np.uint16(0x8691), np.uint16(0x1F68), np.uint16(0xBF45), np.uint16(0xA5D6),
                             np.uint16(0xC295), np.uint16(0xB6F7), np.uint16(0x95D9), np.uint16(0xB9BE)])
        result = blake3_hash(message)
        for i in range(len(expected)):
            self.assertEqual(expected[i], result[i])

    def test_blake3_hash_4(self):
        message = 'a'*48479
        expected = np.array([np.uint16(0x3705), np.uint16(0xB383), np.uint16(0xC5F6), np.uint16(0x199B),
                             np.uint16(0x874D), np.uint16(0xD66A), np.uint16(0x8BB0), np.uint16(0xE749)])
        result = blake3_hash(message)
        for i in range(len(expected)):
            self.assertEqual(expected[i], result[i])

if __name__ == '__main__':
    unittest.main()
