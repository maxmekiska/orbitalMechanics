import numpy as np

class Hohmann:

    def __init__(self, ria, rip, rfa, rfp, planet_rad = 6378, mu = 398600):
        self.ria = ria + planet_rad
        self.rip = rip + planet_rad
        self.rfa = rfa + planet_rad
        self.rfp = rfp + planet_rad
        self.planet_rad = planet_rad
        self.mu = mu
                
        self.h1 = np.sqrt(2*self.mu)*np.sqrt((self.ria*self.rip)/(self.ria + self.rip))
        self.h2 = np.sqrt(2*self.mu)*np.sqrt((self.rfa*self.rip)/(self.rfa + self.rip))
        self.h3 = np.sqrt(2*self.mu)*np.sqrt((self.rfa*self.rfp)/(self.rfa + self.rfp))

        self.va1 = self.h1 / self.rip
        self.va2 = self.h2 / self.rip
        self.va  = self.va2 - self.va1
        self.va  *= 1000

        self.vb2 = self.h2 / self.rfa
        self.vb3 = self.h3 / self.rfa
        self.vb  = self.vb3 - self.vb2
        self.vb  *= 1000

    def delta_v(self):
        return  abs(self.va) + abs(self.vb)

    def delta_m(self, mass, isp, g):
        deltav = self.delta_v()
        deltam = mass * (1 - np.exp(-(deltav /(isp*g))))
        return deltam
