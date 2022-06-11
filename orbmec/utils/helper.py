import numpy as np
from orbmec.utils.transformMatrix import *

def angular_mo(mu, apogee, perigee):
    return np.sqrt( 2 * mu) * np.sqrt((apogee*perigee)/(apogee+perigee))

def QxX(omega, i, w):
    QXx = R_clockwise_yaw(w) @ R_clockwise_roll(i) @ R_clockwise_yaw(omega)
    result = QXx.transpose()
    return result

def r_w(h, mu, e, phi):
    phi = np.deg2rad(phi)

    r = (((h**2)/mu) * (1/(1 + e * np.cos(phi)))) * np.array([[np.cos(phi)],
                                                             [np.sin(phi)],
                                                             [0]])
    return r

def v_w(h, mu, e, phi):
    phi = np.deg2rad(phi)

    v = (mu/h) * np.array([[- np.sin(phi)],
                          [e + np.cos(phi)],
                          [0]])
    return v

def geo_equatorial_frame_position(omega, i, w, h, mu, e, phi):
    position_ = QxX(omega, i, w) @ r_w(h, mu, e, phi)
    return position_

def geo_equatorial_frame_velocity(omega, i, w, h, mu, e, phi):
    velocity_ = QxX(omega, i, w) @ v_w(h, mu, e, phi)
    return velocity_

def orb_elements(r, v, mu):
    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)

    v_r = (np.dot(r,v.T)/r_mag)[0][0]

    h = np.cross(r, v)
    h_mag = np.linalg.norm(h)

    i = np.arccos((h[0][2]/h_mag)) * (180/np.pi)

    n = np.cross(np.array([0, 0, 1]), h)

    n_mag = np.linalg.norm(n)

    if n[0][1] >= 0:
        omega = np.arccos(n[0][0] / n_mag) * (180/np.pi)
    else:
        omega = 360 - np.arccos(n[0][0] / n_mag) * (180/np.pi)

    e = (1 / mu)*((v_mag**2 - (mu / r_mag))*r - r_mag * v_r*v)
    e_mag = np.linalg.norm(e)

    if e[0][2] >= 0:
        w = np.arccos(np.dot(n, e.T) / (n_mag*e_mag)) * (180/np.pi)
    else:
        w = 360 - np.arccos(np.dot(n, e.T) / (n_mag * e_mag)) * (180/np.pi)

    if v_r >= 0:
        theta = np.arccos(np.dot(e/e_mag,(r/r_mag).T)) * (180/np.pi)
    else:
        theta = 360 - np.arccos(np.dot(e/e_mag,(r/r_mag).T)) * (180/np.pi)

    r_p = ((h_mag**2)/mu) * (1/(1 + e_mag * np.cos(np.deg2rad(0))))
    r_a = ((h_mag**2)/mu) * (1/(1 + e_mag * np.cos(np.deg2rad(180))))

    a = 0.5 * (r_p + r_a)

    return (a, i, omega, e_mag, w[0][0], theta[0][0])

def perigee_rad(h_mag, mu, e_mag):
    return ((h_mag**2)/mu) * (1/(1 + e_mag * np.cos(np.deg2rad(0))))

def apogee_rad(h_mag, mu, e_mag):
    return ((h_mag**2)/mu) * (1/(1 + e_mag * np.cos(np.deg2rad(180))))

def angular_momentum(r, v):
    h = np.cross(r, v)
    h_mag = np.linalg.norm(h)
    return h_mag

def period(mu, semimajor):
    return (2*np.pi/np.sqrt(mu))* semimajor**(3/2) * (1/3600)

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
