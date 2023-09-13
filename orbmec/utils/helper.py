import numpy as np

from orbmec.utils.transformMatrix import *


def angular_momentum(mu: float, apogee: float, perigee: float) -> float:
    return np.sqrt(2 * mu) * np.sqrt((apogee * perigee) / (apogee + perigee))


def QXx(omega: float, i: float, w: float) -> np.ndarray:
    return R_clockwise_yaw(w) @ R_clockwise_roll(i) @ R_clockwise_yaw(omega)


def QxX(omega: float, i: float, w: float) -> np.ndarray:
    temp = R_clockwise_yaw(w) @ R_clockwise_roll(i) @ R_clockwise_yaw(omega)
    result = temp.transpose()
    return result


def r_w(h: float, mu: float, e: float, phi: float) -> np.ndarray:
    phi = np.deg2rad(phi)
    r = (((h**2) / mu) * (1 / (1 + e * np.cos(phi)))) * np.array(
        [[np.cos(phi)], [np.sin(phi)], [0]]
    )
    return r


def v_w(h: float, mu: float, e: float, phi: float) -> np.ndarray:
    phi = np.deg2rad(phi)
    v = (mu / h) * np.array([[-np.sin(phi)], [e + np.cos(phi)], [0]])
    return v


def geo_equatorial_frame_position(
    omega: float, i: float, w: float, h: float, mu: float, e: float, phi: float
) -> np.ndarray:
    position_ = QxX(omega, i, w) @ r_w(h, mu, e, phi)
    return position_


def geo_equatorial_frame_velocity(
    omega: float, i: float, w: float, h: float, mu: float, e: float, phi: float
) -> np.ndarray:
    velocity_ = QxX(omega, i, w) @ v_w(h, mu, e, phi)
    return velocity_


def orb_elements(r: np.ndarray, v: np.ndarray, mu: float) -> tuple:
    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)
    v_r = (np.dot(r, v.T) / r_mag)[0][0]
    h = np.cross(r, v)
    h_mag = np.linalg.norm(h)
    i = np.arccos((h[0][2] / h_mag)) * (180 / np.pi)
    n = np.cross(np.array([0, 0, 1]), h)
    n_mag = np.linalg.norm(n)
    if n[0][1] >= 0:
        omega = np.arccos(n[0][0] / n_mag) * (180 / np.pi)
    else:
        omega = 360 - np.arccos(n[0][0] / n_mag) * (180 / np.pi)
    e = (1 / mu) * ((v_mag**2 - (mu / r_mag)) * r - r_mag * v_r * v)
    e_mag = np.linalg.norm(e)
    if e[0][2] >= 0:
        w = np.arccos(np.dot(n, e.T) / (n_mag * e_mag)) * (180 / np.pi)
    else:
        w = 360 - np.arccos(np.dot(n, e.T) / (n_mag * e_mag)) * (180 / np.pi)
    if v_r >= 0:
        theta = np.arccos(np.dot(e / e_mag, (r / r_mag).T)) * (180 / np.pi)
    else:
        theta = 360 - np.arccos(np.dot(e / e_mag, (r / r_mag).T)) * (180 / np.pi)
    r_p = ((h_mag**2) / mu) * (1 / (1 + e_mag * np.cos(np.deg2rad(0))))
    r_a = ((h_mag**2) / mu) * (1 / (1 + e_mag * np.cos(np.deg2rad(180))))
    a = 0.5 * (r_p + r_a)
    return (a, i, omega, e_mag, w[0][0], theta[0][0])


def perigee_rad(h_mag: float, mu: float, e_mag: float) -> float:
    return ((h_mag**2) / mu) * (1 / (1 + e_mag * np.cos(np.deg2rad(0))))


def apogee_rad(h_mag: float, mu: float, e_mag: float) -> float:
    return ((h_mag**2) / mu) * (1 / (1 + e_mag * np.cos(np.deg2rad(180))))


def period(mu: float, semimajor: float) -> float:
    return (2 * np.pi / np.sqrt(mu)) * semimajor ** (3 / 2) * (1 / 3600)
