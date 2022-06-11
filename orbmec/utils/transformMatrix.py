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
