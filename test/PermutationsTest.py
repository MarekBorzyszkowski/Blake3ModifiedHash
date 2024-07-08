import unittest

import numpy as np

from src.Permutations import rotl, G_function, permute_m_by_s


class MyTestCase(unittest.TestCase):
    def test_rol_one_one_shit(self):
        number = np.uint16(1)
        shift = 1
        expected = np.uint16(2)
        self.assertEqual(expected, rotl(number, shift))  # add assertion here

    def test_rol_shift_with_one_going_to_beginning(self):
        number = np.uint16(32769)
        shift = 1
        expected = np.uint16(3)
        self.assertEqual(expected, rotl(number, shift))  # add assertion here

    def test_rol_number_length_rol(self):
        number = np.uint16(1234)
        shift = 16
        expected = np.uint16(1234)
        self.assertEqual(expected, rotl(number, shift))  # add assertion here

    def test_adding_mod_16(self):
        number = np.uint16(65535)
        expected = np.uint16(0)
        self.assertEqual(expected, number + 1)

    def test_G_function(self):
        v00 = np.uint16(0)
        v10 = np.uint16(0)
        v20 = np.uint16(0x03F4)
        v30 = np.uint16(0)
        m0 = np.uint16(0x0001)
        m1 = np.uint16(0x0203)
        v00, v10, v20, v30 = G_function(v00, v10, v20, v30, m0, m1)
        self.assertEqual(np.uint16(0xE223), v00)
        self.assertEqual(np.uint16(0x968D), v10)
        self.assertEqual(np.uint16(0x8CAB), v20)
        self.assertEqual(np.uint16(0x88AF), v30)

    def test_permute_m_by_s(self):
        m = np.array([np.uint16(((2 * i) << 8) + (2 * i + 1)) for i in range(16)])
        expected = np.array([np.uint16(0x0A0B), np.uint16(0x1011), np.uint16(0x0001), np.uint16(0x0405),
                             np.uint16(0x0C0D), np.uint16(0x1617), np.uint16(0x0203), np.uint16(0x0809),
                             np.uint16(0x1E1F), np.uint16(0x1819), np.uint16(0x0607), np.uint16(0x1213),
                             np.uint16(0x1415), np.uint16(0x0E0F), np.uint16(0x1A1B), np.uint16(0x1C1D)])
        m = permute_m_by_s(m)
        for i in range(len(m)):
            self.assertEqual(expected[i], m[i])


if __name__ == '__main__':
    unittest.main()
