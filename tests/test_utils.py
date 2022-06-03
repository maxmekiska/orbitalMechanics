import numpy as np
import unittest


from orbmec.utils.helper import *

class TestUtils(unittest.TestCase):
    omega = 40
    i = 30
    w = 60
    h = 80000
    mu = 398600
    e = 1.4
    phi = 30

    r = np.array([[-6045, -3490, 2500]])
    v = np.array([[-3.457, 6.618, 2.533]])
    mu = 398600

    correct_position = np.array([[-4039.8959232 ], [ 4814.56048018], [ 3628.62470217]])
    correct_velocity = np.array([[-10.38598762], [ -4.77192164], [  1.743875  ]])

    correct_orb_elements = (58311.66993185606, 153.2492285182475, 255.27928533439618, 0.17121234628445342, 20.068316650582467, 28.445628306614996)

    def test_position(self):
        np.testing.assert_allclose(geo_equatorial_frame_position(TestUtils.omega, TestUtils.i, TestUtils.w, TestUtils.h, TestUtils.mu, TestUtils.e, TestUtils.phi), TestUtils.correct_position)

    def test_velocity(self):
        np.testing.assert_allclose(geo_equatorial_frame_velocity(TestUtils.omega, TestUtils.i, TestUtils.w, TestUtils.h, TestUtils.mu, TestUtils.e, TestUtils.phi), TestUtils.correct_velocity)

    def test_orb_elements(self):
        self.assertEqual(orb_elements(TestUtils.r, TestUtils.v, TestUtils.mu), TestUtils.correct_orb_elements)

if __name__ == '__main__':
    unittest.main()
