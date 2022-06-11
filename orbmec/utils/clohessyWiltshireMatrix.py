import numpy as np

def rr(n, t):
    matrix = np.array([[4 - 3 * np.cos(n*t), 0, 0],
                  [6 * (np.sin(n * t) - n* t), 1, 0],
                  [0, 0, np.cos(n * t)]])
    return matrix

def rv(n, t):
    matrix = np.array([[(1/n) * np.sin(n*t), (2/n) * (1 - np.cos(n * t)), 0],
                  [(2/n) * (np.cos(n * t) - 1), (1/n) * (4* np.sin(n*t) - 3*n*t), 0],
                  [0, 0, (1/n) * np.sin(n * t)]])
    return matrix

def vr(n, t):
    matrix = np.array([[3 * n * np.sin(n * t), 0, 0],
                  [6 * n * (np.cos(n * t) -1 ), 0, 0],
                  [0, 0, -n * np.sin(n * t)]])
    return matrix

def vv(n, t):
    matrix = np.array([[np.cos(n * t), 2 * np.sin(n * t), 0],
                  [-2 * np.sin(n * t), 4 * np.cos(n * t) - 3, 0],
                  [0, 0, np.cos(n * t)]])
    return matrix
