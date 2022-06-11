import numpy as np
from orbmec.utils.helper import angular_mo

class Phasing:

    def __init__(self, ria, rip, target_theta, n = 1, mu = 398600):
        self.ria = ria
        self.rip = rip
        self.target_theta = np.deg2rad(target_theta)
        self.mu = mu
        self.n = n

        self.h1 = angular_mo(mu, ria, rip)

        self.a1  = (1/2) * (ria+rip);
        self.t1  = ( (2*np.pi)/np.sqrt(mu) )*( self.a1**(3/2) )
        self.e = (ria-rip) / (ria+rip)

        self.Eb = 2 * np.arctan( np.sqrt( (1-self.e)/(1+self.e) ) * np.tan(self.target_theta/2) )


        self.tab = (self.t1/(2*np.pi)) * (self.Eb - self.e*np.sin(self.Eb))

        self.t2 = self.t1 - (self.tab/self.n)

        self.a2 = ( (np.sqrt(mu)*self.t2)/(2*np.pi) )**(2/3)

        self.rpa = 2*self.a2 - rip
        self.h2 = angular_mo(mu, rip, self.rpa)

        self.va1 = self.h1/rip
        self.va2 = self.h2/rip

    @property
    def apogee_phasing_orbit(self):
        return self.rpa

    def delta_v(self):
        return abs(self.va1 - self.va2) + abs(self.va2 - self.va1)
