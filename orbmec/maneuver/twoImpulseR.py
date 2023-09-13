import numpy as np

from orbmec.utils.clohessyWiltshireMatrix import *


class TwoImpulseR:
    def __init__(
        self,
        position_target: np.ndarray,
        velocity_target: np.ndarray,
        position_chaser: np.ndarray,
        velocity_chaser: np.ndarray,
        t: float,
    ):
        self.position_target: np.ndarray = position_target
        self.velocity_target: np.ndarray = velocity_target
        self.position_chaser: np.ndarray = position_chaser
        self.velocity_chaser: np.ndarray = velocity_chaser
        self.t: float = t

        self.i: np.ndarray = position_target / np.linalg.norm(position_target)
        self.j: np.ndarray = velocity_target / np.linalg.norm(velocity_target)
        self.k: np.ndarray = np.cross(self.i, self.j)

        self.QXx: np.ndarray = np.vstack((self.i, self.j, self.k))

        self.n: float = np.linalg.norm(velocity_target) / np.linalg.norm(
            position_target
        )
        self.Wss: np.ndarray = self.n * self.k

        self.dr: np.ndarray = self.position_chaser - self.position_target
        self.dv: np.ndarray = (
            self.velocity_chaser - self.velocity_target - np.cross(self.Wss, self.dr)
        )

        self.dr0: np.ndarray = self.QXx @ self.dr.reshape(-1, 1)
        self.dv0_: np.ndarray = self.QXx @ self.dv.reshape(-1, 1)

        self.phi_rr: np.ndarray = rr(self.n, self.t)
        self.phi_rv: np.ndarray = rv(self.n, self.t)
        self.phi_vr: np.ndarray = vr(self.n, self.t)
        self.phi_vv: np.ndarray = vv(self.n, self.t)

        self.dv0p: np.ndarray = -np.linalg.inv(self.phi_rv) @ self.phi_rr @ self.dr0
        self.dvf_: np.ndarray = (self.phi_vr @ self.dr0) + self.phi_vv @ self.dv0p

    @property
    def QXx_value(self) -> np.ndarray:
        return self.QXx

    @property
    def n_value(self) -> float:
        return self.n

    @property
    def Wss_value(self) -> np.ndarray:
        return self.Wss

    @property
    def dr_value(self) -> np.ndarray:
        return self.dr

    @property
    def dv_value(self) -> np.ndarray:
        return self.dv

    @property
    def dr0_value(self) -> np.ndarray:
        return self.dr0

    @property
    def dv0_value(self) -> np.ndarray:
        return self.dv0_

    @property
    def phi_rr_value(self) -> np.ndarray:
        return self.phi_rr

    @property
    def phi_rv_value(self) -> np.ndarray:
        return self.phi_rv

    @property
    def phi_vr_value(self) -> np.ndarray:
        return self.phi_vr

    @property
    def phi_vv_value(self) -> np.ndarray:
        return self.phi_vv

    @property
    def dv0p_value(self) -> np.ndarray:
        return self.dv0p

    @property
    def dvf_value(self) -> np.ndarray:
        return self.dvf_

    def deltav_start(self) -> np.ndarray:
        return self.dv0p - self.dv0_

    def deltav_end(self) -> np.ndarray:
        return np.zeros(shape=(3, 1)) - self.dvf_

    def totaldeltav(self) -> float:
        delta_v0 = self.deltav_start()
        delta_vf = self.deltav_end()
        return (np.linalg.norm(delta_v0) + np.linalg.norm(delta_vf)) * 1000

    def relative_position(self, t: float) -> np.ndarray:
        term1 = rr(self.n, t) @ self.dr0
        term2 = rv(self.n, t) @ self.dv0p

        return term1 + term2
