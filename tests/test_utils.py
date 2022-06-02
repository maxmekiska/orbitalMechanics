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
    correct_position = np.array([[-4039.8959232 ], [ 4814.56048018], [ 3628.62470217]])
    correct_velocity = np.array([[-10.38598762], [ -4.77192164], [  1.743875  ]])

    def test_position(self):
        np.testing.assert_allclose(position(TestUtils.omega, TestUtils.i, TestUtils.w, TestUtils.h, TestUtils.mu, TestUtils.e, TestUtils.phi), TestUtils.correct_position)

    def test_velocity(self):
        np.testing.assert_allclose(velocity(TestUtils.omega, TestUtils.i, TestUtils.w, TestUtils.h, TestUtils.mu, TestUtils.e, TestUtils.phi), TestUtils.correct_velocity)

if __name__ == '__main__':
    unittest.main()
