class TwoImpulseR:

    def __init__(self, position_target, velocity_target, position_chaser, velocity_chaser, t):
        self.position_target = position_target
        self.velocity_target = velocity_target
        self.position_chaser = position_chaser
        self.velocity_chaser = velocity_chaser
        self.time = t

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

        self.phi_rr = self.__rr(n, t)
        self.phi_rv = self.__rv(n, t)
        self.phi_vr = self.__vr(n, t)
        self.phi_vv = self.__vv(n, t)

        self.dv0p = - np.linalg.inv(self.phi_rv) @ self.phi_rr @ self.dr0
        self.dvf_ = (self.phi_vr @ self.dr0) + self.phi_vv @ self.dv0p

    def __rr(self, n, t):
        matrix = np.array([[4 - 3 * np.cos(n*t), 0, 0],
                      [6 * (np.sin(n * t) - n* t), 1, 0],
                      [0, 0, np.cos(n * t)]])
        return matrix

    def __rv(self, n, t):
        matrix = np.array([[(1/n) * np.sin(n*t), (2/n) * (1 - np.cos(n * t)), 0],
                      [(2/n) * (np.cos(n * t) - 1), (1/n) * (4* np.sin(n*t) - 3*n*t), 0],
                      [0, 0, (1/n) * np.sin(n * t)]])
        return matrix

    def __vr(self, n, t):
        matrix = np.array([[3 * n * np.sin(n * t), 0, 0],
                      [6 * n * (np.cos(n * t) -1 ), 0, 0],
                      [0, 0, -n * np.sin(n * t)]])
        return matrix

    def __vv(self, n, t):
        matrix = np.array([[np.cos(n * t), 2 * np.sin(n * t), 0],
                      [-2 * np.sin(n * t), 4 * np.cos(n * t) - 3, 0],
                      [0, 0, np.cos(n * t)]])
        return matrix

    def deltav_start(self):
        return  self.dv0p - self.dv0_

    def deltav_end(self):
        return np.zeros(shape=(3, 1)) - self.dvf_

    def totaldeltav(self):
        delta_v0 = self.deltav_start()
        delta_vf = self.deltav_end()
        return (np.linalg.norm(delta_v0) + np.linalg.norm(delta_vf)) * 1000

    def relative_position(self, t):
        term1 = self.__rr(self.n, t) @ self.dr0
        term2 = self.__rv(self.n, t) @ self.dv0p

        return term1 + term2
