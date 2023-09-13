import unittest

import numpy as np

from orbmec.maneuver.hohmann import *
from orbmec.maneuver.phasing import *
from orbmec.maneuver.twoImpulseR import *


class TestManeuver(unittest.TestCase):

    position_target = np.array([1622.39, 5305.10, 3717.44])
    velocity_target = np.array([-7.29936, 0.492329, 2.48304])
    position_chaser = np.array([1612.75, 5310.19, 3750.33])
    velocity_chaser = np.array([-7.35170, 0.463828, 2.46906])
    t = 28800

    test_obj = TwoImpulseR(
        position_target, velocity_target, position_chaser, velocity_chaser, t
    )
    test_obj2 = Hohmann(800, 480, 16000, 16000)
    test_obj3 = Phasing(mu=398600, n=1, ria=13600, rip=6800, target_theta=90)

    correct_deltav_start = np.array([[0.02930887], [-0.06676299], [0.0129857]])
    correct_deltav_end = np.array([[0.02581208], [0.00047124], [0.02447875]])
    correct_relative_position = np.array([[20.02902597], [19.90929503], [20.0173155]])
    correct_totaldeltav = 109.6369494855627
    correct_deltav = 3052.2018535720463
    correct_deltam = 1291.2664959148321

    correct_deltav_phasing = 0.49702295194280666
    correct_apogee_phasing_orbit = 11564.147485565016

    def test_deltav_start(self):
        np.testing.assert_allclose(
            TestManeuver.test_obj.deltav_start(),
            TestManeuver.correct_deltav_start,
            rtol=0.005,
            atol=0.005,
        )

    def test_deltav_end(self):
        np.testing.assert_allclose(
            TestManeuver.test_obj.deltav_end(),
            TestManeuver.correct_deltav_end,
            rtol=0.005,
            atol=0.005,
        )

    def test_relative_position(self):
        np.testing.assert_allclose(
            TestManeuver.test_obj.relative_position(t=2),
            TestManeuver.correct_relative_position,
            rtol=0.005,
            atol=0.005,
        )

    def test_totaldeltav(self):
        np.testing.assert_allclose(
            TestManeuver.test_obj.totaldeltav(),
            TestManeuver.correct_totaldeltav,
            rtol=0.005,
            atol=0.005,
        )

    def test_delta_v(self):
        self.assertEqual(TestManeuver.test_obj2.delta_v(), TestManeuver.correct_deltav)

    def test_delta_m(self):
        self.assertEqual(
            TestManeuver.test_obj2.delta_m(2000, 300, 9.807),
            TestManeuver.correct_deltam,
        )

    def test_delta_v_pahsing(self):
        self.assertEqual(
            TestManeuver.test_obj3.delta_v(), TestManeuver.correct_deltav_phasing
        )

    def test_apogee_target_orbit(self):
        self.assertEqual(
            TestManeuver.test_obj3.apogee_phasing_orbit,
            TestManeuver.correct_apogee_phasing_orbit,
        )


if __name__ == "__main__":
    unittest.main()
