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

def position(omega, i, w, h, mu, e, phi):
    position_ = QxX(omega, i, w) @ r_w(h, mu, e, phi)
    return position_

def velocity(omega, i, w, h, mu, e, phi):
    velocity_ = QxX(omega, i, w) @ v_w(h, mu, e, phi)
    return velocity_
