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

    h_mag = 58311.66993185606
    e_mag = 0.17121234628445342
    perigee = 7283.464732960478
    apogee = 10292.725501794834
    semimajor = 8788.095117377656

    correct_position = np.array([[-4039.8959232 ], [ 4814.56048018], [ 3628.62470217]])
    correct_velocity = np.array([[-10.38598762], [ -4.77192164], [  1.743875  ]])

    correct_orb_elements = (58311.66993185606, 153.2492285182475, 255.27928533439618, 0.17121234628445342, 20.068316650582467, 28.445628306614996)
    correct_perigee = 7283.464732960478
    correct_apogee = 10292.725501794834
    correct_semimajor = 8788.095117377656
    correct_period = 2.2774604491192245

    def test_position(self):
        np.testing.assert_allclose(geo_equatorial_frame_position(TestUtils.omega, TestUtils.i, TestUtils.w, TestUtils.h, TestUtils.mu, TestUtils.e, TestUtils.phi), TestUtils.correct_position)

    def test_velocity(self):
        np.testing.assert_allclose(geo_equatorial_frame_velocity(TestUtils.omega, TestUtils.i, TestUtils.w, TestUtils.h, TestUtils.mu, TestUtils.e, TestUtils.phi), TestUtils.correct_velocity)

    def test_orb_elements(self):
        self.assertEqual(orb_elements(TestUtils.r, TestUtils.v, TestUtils.mu), TestUtils.correct_orb_elements)

    def test_perigee_rad(self):
        self.assertEqual(perigee_rad(TestUtils.h_mag, TestUtils.mu, TestUtils.e_mag), TestUtils.correct_perigee)

    def test_apogee_rad(self):
        self.assertEqual(apogee_rad(TestUtils.h_mag, TestUtils.mu, TestUtils.e_mag), TestUtils.correct_apogee)

    def test_semimajor(self):
        self.assertEqual(semimajor(TestUtils.perigee, TestUtils.apogee), TestUtils.correct_semimajor)

    def test_period(self):
        self.assertEqual(period(TestUtils.mu, TestUtils.semimajor), TestUtils.correct_period)

if __name__ == '__main__':
    unittest.main()
