import numpy as np

from orbmec.utils.helper import angular_momentum


class Phasing:
    def __init__(
        self,
        ria: float,
        rip: float,
        target_theta: float,
        n: int = 1,
        mu: float = 398600,
    ):
        self.ria: float = ria
        self.rip: float = rip
        self.target_theta: float = np.deg2rad(target_theta)
        self.mu: float = mu
        self.n: int = n

        self.h1: float = angular_momentum(mu, ria, rip)

        self.a1: float = (1 / 2) * (ria + rip)
        self.t1: float = ((2 * np.pi) / np.sqrt(mu)) * (self.a1 ** (3 / 2))
        self.e: float = (ria - rip) / (ria + rip)

        self.Eb: float = 2 * np.arctan(
            np.sqrt((1 - self.e) / (1 + self.e)) * np.tan(self.target_theta / 2)
        )

        self.tab: float = (self.t1 / (2 * np.pi)) * (self.Eb - self.e * np.sin(self.Eb))

        self.t2: float = self.t1 - (self.tab / self.n)

        self.a2: float = ((np.sqrt(mu) * self.t2) / (2 * np.pi)) ** (2 / 3)

        self.rpa: float = 2 * self.a2 - rip
        self.h2: float = angular_momentum(mu, rip, self.rpa)

        self.va1: float = self.h1 / rip
        self.va2: float = self.h2 / rip

    @property
    def apogee_phasing_orbit(self) -> float:
        return self.rpa

    def delta_v(self) -> float:
        return abs(self.va1 - self.va2) + abs(self.va2 - self.va1)
