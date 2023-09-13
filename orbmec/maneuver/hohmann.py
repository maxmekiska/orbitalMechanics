import numpy as np

from orbmec.utils.helper import angular_momentum


class Hohmann:
    def __init__(
        self,
        ria: float,
        rip: float,
        rfa: float,
        rfp: float,
        planet_rad: float = 6378,
        mu: float = 398600,
    ):
        self.ria: float = ria + planet_rad
        self.rip: float = rip + planet_rad
        self.rfa: float = rfa + planet_rad
        self.rfp: float = rfp + planet_rad
        self.planet_rad: float = planet_rad
        self.mu: float = mu

        self.h1: float = angular_momentum(mu, self.ria, self.rip)
        self.h2: float = angular_momentum(mu, self.rfa, self.rip)
        self.h3: float = angular_momentum(mu, self.rfa, self.rfp)

        self.va1: float = self.h1 / self.rip
        self.va2: float = self.h2 / self.rip
        self.va: float = self.va2 - self.va1
        self.va *= 1000  # Converting to m/s

        self.vb2: float = self.h2 / self.rfa
        self.vb3: float = self.h3 / self.rfa
        self.vb: float = self.vb3 - self.vb2
        self.vb *= 1000  # Converting to m/s

    def delta_v(self) -> float:
        return abs(self.va) + abs(self.vb)

    def delta_m(self, mass: float, isp: float, g: float) -> float:
        deltav: float = self.delta_v()
        deltam: float = mass * (1 - np.exp(-(deltav / (isp * g))))
        return deltam
