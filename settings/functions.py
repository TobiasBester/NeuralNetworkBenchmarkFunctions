import numpy as np


def ackley_n2_func(x, y):
    return -200 * np.exp(-0.2 * np.sqrt(x ** 2 + y ** 2))


def ackley_n3_func(x, y):
    return ackley_n2_func(x, y) + (5 * np.exp(np.cos(3 * x) + np.sin(3 * y)))


def adjiman_func(x, y):
    return np.cos(x) * np.sin(y) - (x / (y**2 + 1))


def beale_func(x, y):
    return (1.5 - x + (x * y)) ** 2 + (2.25 - x + (x * y ** 2)) ** 2 + (2.625 - x + (x * y ** 3)) ** 2


def bartels_conn_func(x, y):
    return np.abs(x ** 2 + y ** 2 + (x * y)) + np.abs(np.sin(x)) + np.abs(np.cos(y))


def bird(x, y):
    return (np.sin(x) * np.exp((1 - np.cos(y)) ** 2)) \
           + (np.cos(y) * np.exp((1 - np.sin(x)) ** 2)) \
           + (x - y) ** 2


def bohachevsky_n1(x, y):
    return (x ** 2) + (2 * y ** 2) - (0.3 * np.cos(3 * np.pi * x)) \
        - (0.4 * np.cos(4 * np.pi * y)) + 0.7


def bohachevsky_n2(x, y):
    return (x ** 2) + (2 * y ** 2) \
           - (0.3 * np.cos(3 * np.pi * x) * np.cos(4 * np.pi * y)) + 0.3


def booth(x, y):
    return (x + (2 * y) - 7) ** 2 + ((2 * x) + y - 5) ** 2


def brent(x, y):
    return (x + 10) ** 2 + (y + 10) ** 2 + (np.exp(-(x ** 2) - (y ** 2)))
