import numpy as np
from .helper import orb_elements

def gibbs(r1, r2, r3, mu, tolerance = 0.0001):
    r1_mag = np.linalg.norm(r1)
    r2_mag = np.linalg.norm(r2)
    r3_mag = np.linalg.norm(r3)

    c12 = np.cross(r1, r2)
    c23 = np.cross(r2, r3)
    c31 = np.cross(r3, r1)

    u_r1 = r1/r1_mag
    u_r2 = r2/r2_mag
    u_r3 = r3/r3_mag

    c23_hat = c23 / np.linalg.norm(c23)

    if (abs(np.dot(c23_hat, u_r1.T))) > tolerance:
        raise ValueError("Not coplanar vectors")
    else:
        pass

    n = r1_mag * c23 + r2_mag * c31 + r3_mag * c12
    n_mag = np.linalg.norm(n)

    d = c12 + c23 + c31
    d_mag = np.linalg.norm(d)

    s = r1 * (r2_mag - r3_mag) + r2 * (r3_mag - r1_mag) + r3 * (r1_mag - r2_mag)

    v2 = np.sqrt(mu/(n_mag*d_mag)) * (((np.cross(d, r2)) / r2_mag) + s)

    result = orb_elements(r2, v2, mu)

    return result
