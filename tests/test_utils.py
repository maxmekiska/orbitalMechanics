import numpy as np
import unittest


from orbmec.utils.helper import *
from orbmec.utils.rkf45 import rkf45
from orbmec.utils.gibbs import gibbs

class TestUtils(unittest.TestCase):
    omega = 40
    i = 30
    w = 60
    h = 80000
    mu = 398600
    e = 1.4
    phi = 30
    ria = 13600
    rip = 6800

    r = np.array([[-6045, -3490, 2500]])
    v = np.array([[-3.457, 6.618, 2.533]])
    r1 = np.array([[-294.32, 4265.1, 5986.7]])
    r2 = np.array([[-1365.5, 3637.6, 6346.8]])
    r3 = np.array([[-2940.3, 2473.7, 6555.8]])
    mu = 398600

    h_mag = 58311.66993185606
    e_mag = 0.17121234628445342
    perigee = 7283.464732960478
    apogee = 10292.725501794834
    semimajor = 8788.095117377656

    correct_position = np.array([[-4039.8959232 ], [ 4814.56048018], [ 3628.62470217]])
    correct_velocity = np.array([[-10.38598762], [ -4.77192164], [  1.743875  ]])

    correct_orb_elements = (8788.095117377656, 153.2492285182475, 255.27928533439618, 0.17121234628445342, 20.068316650582467, 28.445628306614996)
    correct_perigee = 7283.464732960478
    correct_apogee = 10292.725501794834
    correct_semimajor = 8788.095117377656
    correct_period = 2.2774604491192245
    correct_angular_momentum = 60116.33166896774
    correct_gibbs = (8001.43789952298, 60.000470277369566, 40.00144177286777, 0.10010369281339095, 30.074116831548118, 49.92565926551749)

    correct_V_X = np.array([[0.        , 0.2       , 0.        , 0.3       ],
                            [0.02472456, 0.19598053, 0.03673761, 0.28746129],
                            [0.04924425, 0.18881162, 0.07198215, 0.26477623],
                            [0.07313022, 0.17846041, 0.10436822, 0.23242569],
                            [0.09352763, 0.16656505, 0.12973534, 0.1961517 ]])

    correct_V_T = np.array([0.         , 0.12471833, 0.25199507, 0.38189235, 0.5])

    def V_test(u,t):
        w = 1
        b = 0.1
        x1,dx1, x2, dx2=u
        ddx1=-w**2 * x1 -b * dx1
        ddx2=-(w+0.5)**2 * x2 -(b+0.1) * dx2
        return np.array([dx1,ddx1,dx2,ddx2])

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

    def test_angular_momentum(self):
        self.assertEqual(angular_momentum(TestUtils.mu, TestUtils.ria, TestUtils.rip), TestUtils.correct_angular_momentum)

    def test_period(self):
        self.assertEqual(period(TestUtils.mu, TestUtils.semimajor), TestUtils.correct_period)

    def test_gibbs(self):
        self.assertEqual(gibbs(TestUtils.r1, TestUtils.r2, TestUtils.r3, TestUtils.mu), TestUtils.correct_gibbs)

    def test_rkf45(self):
        np.testing.assert_allclose(rkf45( f=TestUtils.V_test, a=0, b=0.5, x0=[0,0.2,0,0.3], tol=1e-6, hmax=1e1, hmin=1e-16 )[0], TestUtils.correct_V_T)
        np.testing.assert_allclose(rkf45( f=TestUtils.V_test, a=0, b=0.5, x0=[0,0.2,0,0.3], tol=1e-6, hmax=1e1, hmin=1e-16 )[1], TestUtils.correct_V_X)

if __name__ == '__main__':
    unittest.main()
