import numpy as np
from orbmec.utils.clohessyWiltshireMatrix import *

class TwoImpulseR:

    def __init__(self, position_target, velocity_target, position_chaser, velocity_chaser, t):
        self.position_target = position_target
        self.velocity_target = velocity_target
        self.position_chaser = position_chaser
        self.velocity_chaser = velocity_chaser
        self.t = t

        self.i = position_target / np.linalg.norm(position_target)
        self.j = velocity_target / np.linalg.norm(velocity_target)
        self.k = np.cross(self.i, self.j)

        self.QXx = np.vstack((self.i, self.j, self.k))

        self.n = np.linalg.norm(velocity_target) / np.linalg.norm(position_target)
        self.Wss = self.n * self.k

        self.dr = self.position_chaser - self.position_target
        self.dv = self.velocity_chaser - self.velocity_target - np.cross(self.Wss, self.dr)

        self.dr0 = self.QXx @ self.dr.reshape(-1, 1)
        self.dv0_ = self.QXx @ self.dv.reshape(-1, 1)

        self.phi_rr = rr(self.n, self.t)
        self.phi_rv = rv(self.n, self.t)
        self.phi_vr = vr(self.n, self.t)
        self.phi_vv = vv(self.n, self.t)

        self.dv0p = - np.linalg.inv(self.phi_rv) @ self.phi_rr @ self.dr0
        self.dvf_ = (self.phi_vr @ self.dr0) + self.phi_vv @ self.dv0p

    @property
    def QXx_value(self):
        return self.QXx

    @property
    def n_value(self):
        return self.n

    @property
    def Wss_value(self):
        return Wss

    @property
    def dr_value(self):
        return self.dr

    @property
    def dv_value(self):
        return self.drv

    @property
    def dr0_value(self):
        return self.dr0

    @property
    def dv0_value(self):
        return self.dv0_

    @property
    def phi_rr_value(self):
        return self.phi_rr

    @property
    def phi_rv_value(self):
        return self.phi_rv

    @property
    def phi_vr_value(self):
        return self.phi_vr

    @property
    def phi_vv_value(self):
        return self.phi_rr

    @property
    def dv0p_value(self):
        return self.dv0p

    @property
    def dvf_value(self):
        return self.dvf_

    def deltav_start(self):
        return  self.dv0p - self.dv0_

    def deltav_end(self):
        return np.zeros(shape=(3, 1)) - self.dvf_

    def totaldeltav(self):
        delta_v0 = self.deltav_start()
        delta_vf = self.deltav_end()
        return (np.linalg.norm(delta_v0) + np.linalg.norm(delta_vf)) * 1000

    def relative_position(self, t):
        term1 = rr(self.n, t) @ self.dr0
        term2 = rv(self.n, t) @ self.dv0p

        return term1 + term2
