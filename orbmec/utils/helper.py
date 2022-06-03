import numpy as np

def R_clockwise_yaw(alpha):
    alpha = np.deg2rad(alpha)
    matrix = np.array([[np.cos(alpha), np.sin(alpha),  0],
                       [-np.sin(alpha), np.cos(alpha), 0],
                       [0,              0,             1]])
    return matrix

def R_counterclockwise_yaw(alpha):
    alpha = np.deg2rad(alpha)
    matrix = np.array([[np.cos(alpha), -np.sin(alpha),  0],
                       [np.sin(alpha), np.cos(alpha),   0],
                       [0,              0,              1]])
    return matrix

def R_clockwise_pitch(beta):
    beta = np.deg2rad(beta)
    matrix = np.array([[np.cos(beta),   0,  -np.sin(beta)],
                       [0,              1,              0],
                       [np.sin(beta),   0,   np.cos(beta)]])
    return matrix

def R_counterclockwise_pitch(beta):
    beta = np.deg2rad(beta)
    matrix = np.array([[np.cos(beta),   0,   np.sin(beta)],
                       [0,              1,              0],
                       [-np.sin(beta),  0,   np.cos(beta)]])
    return matrix

def R_clockwise_roll(gamma):
    gamma = np.deg2rad(gamma)
    matrix = np.array([[1,              0,              0],
                       [0,  np.cos(gamma),  np.sin(gamma)],
                       [0, -np.sin(gamma),  np.cos(gamma)]])
    return matrix

def R_counterclockwise_roll(gamma):
    gamma = np.deg2rad(gamma)
    matrix = np.array([[1,              0,              0],
                       [0,  np.cos(gamma), -np.sin(gamma)],
                       [0, np.sin(gamma),   np.cos(gamma)]])
    return matrix

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

    return (h_mag, i, omega, e_mag, w[0][0], theta[0][0])

def perigee_rad(h_mag, mu, e_mag):
    return ((h_mag**2)/mu) * (1/(1 + e_mag * np.cos(np.deg2rad(0))))

def apogee_rad(h_mag, mu, e_mag):
    return ((h_mag**2)/mu) * (1/(1 + e_mag * np.cos(np.deg2rad(180))))

def semimajor(perigee, apogee):
    return 0.5 * (perigee + apogee)

def period(mu, semimajor):
    return (2*np.pi/np.sqrt(mu))* semimajor**(3/2) * (1/3600)
