"""
source:
https://stackoverflow.com/questions/65416794/issue-on-runge-kutta-fehlberg-algorithm
"""

from typing import Callable, Tuple

import numpy as np


def rkf45(
    f: Callable[[np.ndarray, float], np.ndarray],
    a: float,
    b: float,
    x0: np.ndarray,
    tol: float,
    hmax: float,
    hmin: float,
) -> Tuple[np.ndarray, np.ndarray]:
    a2 = 2.500000000000000e-01
    a3 = 3.750000000000000e-01
    a4 = 9.230769230769231e-01
    a5 = 1.000000000000000e00
    a6 = 5.000000000000000e-01

    b21 = 2.500000000000000e-01
    b31 = 9.375000000000000e-02
    b32 = 2.812500000000000e-01
    b41 = 8.793809740555303e-01
    b42 = -3.277196176604461e00
    b43 = 3.320892125625853e00
    b51 = 2.032407407407407e00
    b52 = -8.000000000000000e00
    b53 = 7.173489278752436e00
    b54 = -2.058966861598441e-01
    b61 = -2.962962962962963e-01
    b62 = 2.000000000000000e00
    b63 = -1.381676413255361e00
    b64 = 4.529727095516569e-01
    b65 = -2.750000000000000e-01

    r1 = 2.777777777777778e-03
    r3 = -2.994152046783626e-02
    r4 = -2.919989367357789e-02
    r5 = 2.000000000000000e-02
    r6 = 3.636363636363636e-02

    c1 = 1.157407407407407e-01
    c3 = 5.489278752436647e-01
    c4 = 5.353313840155945e-01
    c5 = -2.000000000000000e-01

    t = a
    x = np.array(x0)
    h = hmax

    T = np.array([t])
    X = np.array([x])

    while t < b:

        if t + h > b:
            h = b - t

        k1 = h * f(x, t)
        k2 = h * f(x + b21 * k1, t + a2 * h)
        k3 = h * f(x + b31 * k1 + b32 * k2, t + a3 * h)
        k4 = h * f(x + b41 * k1 + b42 * k2 + b43 * k3, t + a4 * h)
        k5 = h * f(x + b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4, t + a5 * h)
        k6 = h * f(x + b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5, t + a6 * h)

        r = abs(r1 * k1 + r3 * k3 + r4 * k4 + r5 * k5 + r6 * k6) / h
        if len(np.shape(r)) > 0:
            r = max(r)
        if r <= tol:
            t = t + h
            x = x + c1 * k1 + c3 * k3 + c4 * k4 + c5 * k5
            T = np.append(T, t)
            X = np.append(X, [x], 0)

        h = h * min(max(0.84 * (tol / r) ** 0.25, 0.1), 4.0)

        if h > hmax:
            h = hmax
        elif h < hmin:
            raise RuntimeError("Error: Could not converge to the required tolerance")

    return (T, X)
