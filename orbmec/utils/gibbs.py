import numpy as np

from .helper import orb_elements


def gibbs(
    r1: np.ndarray, r2: np.ndarray, r3: np.ndarray, mu: float, tolerance: float = 0.0001
) -> dict:
    r1_mag: float = np.linalg.norm(r1)
    r2_mag: float = np.linalg.norm(r2)
    r3_mag: float = np.linalg.norm(r3)

    c12: np.ndarray = np.cross(r1, r2)
    c23: np.ndarray = np.cross(r2, r3)
    c31: np.ndarray = np.cross(r3, r1)

    u_r1: np.ndarray = r1 / r1_mag
    u_r2: np.ndarray = r2 / r2_mag
    u_r3: np.ndarray = r3 / r3_mag

    c23_hat: np.ndarray = c23 / np.linalg.norm(c23)

    if (abs(np.dot(c23_hat, u_r1.T))) > tolerance:
        raise ValueError("Not coplanar vectors")
    else:
        pass

    n: np.ndarray = r1_mag * c23 + r2_mag * c31 + r3_mag * c12
    n_mag: float = np.linalg.norm(n)

    d: np.ndarray = c12 + c23 + c31
    d_mag: float = np.linalg.norm(d)

    s: np.ndarray = (
        r1 * (r2_mag - r3_mag) + r2 * (r3_mag - r1_mag) + r3 * (r1_mag - r2_mag)
    )

    v2: np.ndarray = np.sqrt(mu / (n_mag * d_mag)) * (((np.cross(d, r2)) / r2_mag) + s)

    result: dict = orb_elements(r2, v2, mu)

    return result
