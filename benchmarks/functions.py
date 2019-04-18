import math
import numpy as np


def beale_func(x, y):
    return (1.5 - x + x * y) ** 2 + (2.25 - x + x * y ** 2) ** 2 + (2.625 - x + x * y ** 3) ** 2


def ackley_n2_func(x, y):
    return -200 * np.exp(-0.2 * np.sqrt(x**2 + y**2))


def adjiman_func(x, y):
    return np.cos(x) * np.sin(y) - (x / (y**2 + 1))
