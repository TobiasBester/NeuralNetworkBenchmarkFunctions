import numpy as np

import settings.functions


def ackley_n2():
    x_range = (-32, 32)
    y_range = x_range
    func = settings.functions.ackley_n2_func
    return func, x_range, y_range, "Ackley N.2"


def ackley_n3():
    x_range = (-32, 32)
    y_range = x_range
    func = settings.functions.ackley_n3_func
    return func, x_range, y_range, "Ackley N.3"


def adjiman():
    x_range = (-1, 2)
    y_range = (-1, 1)
    func = settings.functions.adjiman_func
    return func, x_range, y_range, "Adjiman"


def beale():
    x_range = (-4.5, 4.5)
    y_range = x_range
    func = settings.functions.beale_func
    return func, x_range, y_range, "Beale"


def bartels_conn():
    x_range = (-500, 500)
    y_range = x_range
    func = settings.functions.bartels_conn_func
    return func, x_range, y_range, "Bartels Conn"


def bird():
    x_range = (-2 * np.pi, 2 * np.pi)
    y_range = x_range
    func = settings.functions.bird
    return func, x_range, y_range, "Bird"


def boha_n1():
    x_range = (-100, 100)
    y_range = x_range
    func = settings.functions.bohachevsky_n1
    return func, x_range, y_range, "Bohachevsky N.1"


def boha_n2():
    x_range = (-100, 100)
    y_range = x_range
    func = settings.functions.bohachevsky_n2
    return func, x_range, y_range, "Bohachevsky N.2"


def booth():
    x_range = (-10, 10)
    y_range = x_range
    func = settings.functions.booth
    return func, x_range, y_range, "Booth"


def brent():
    x_range = (-20, 0)
    y_range = x_range
    func = settings.functions.brent
    return func, x_range, y_range, "Brent"
