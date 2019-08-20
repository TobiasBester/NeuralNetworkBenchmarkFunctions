import numpy as np

import settings.functions


def ackley():
    x_range = (-32, 32)
    y_range = x_range
    func = settings.functions.ackley_func
    return func, x_range, y_range, "Ackley"


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


def ackley_n4():
    x_range = (-35, 35)
    y_range = x_range
    func = settings.functions.ackley_n4_func
    return func, x_range, y_range, "Ackley N.4"


def adjiman():
    x_range = (-1, 2)
    y_range = (-1, 1)
    func = settings.functions.adjiman_func
    return func, x_range, y_range, "Adjiman"


def alpine_n1():
    x_range = (0, 10)
    y_range = x_range
    func = settings.functions.alpine_n1_func
    return func, x_range, y_range, "Alpine N.1"


def alpine_n2():
    x_range = (0, 10)
    y_range = x_range
    func = settings.functions.alpine_n2_func
    return func, x_range, y_range, "Alpine N.2"


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


def brown():
    x_range = (-1, 1)
    y_range = x_range
    func = settings.functions.brown
    return func, x_range, y_range, "Brown"


def bukin_n6():
    x_range = (-15, -5)
    y_range = (-3, 3)
    func = settings.functions.bukin_n6
    return func, x_range, y_range, "Bukin N.6"


def cross_in_tray():
    x_range = (-10, 10)
    y_range = x_range
    func = settings.functions.cross_in_tray
    return func, x_range, y_range, "Cross-In-Tray"


def deckkers_arts():
    x_range = (-20, 20)
    y_range = x_range
    func = settings.functions.deckkers_aarts
    return func, x_range, y_range, "Deckkers-Aarts"


def drop_wave():
    x_range = (-5.2, 5.2)
    y_range = x_range
    func = settings.functions.drop_wave
    return func, x_range, y_range, "Drop-Wave"


def easom():
    x_range = (-50, 50)
    y_range = x_range
    func = settings.functions.easom
    return func, x_range, y_range, "Easom"


def egg_crate():
    x_range = (-5, 5)
    y_range = x_range
    func = settings.functions.egg_crate
    return func, x_range, y_range, "Egg Crate"


def exponential():
    x_range = (-1, 1)
    y_range = x_range
    func = settings.functions.exponential
    return func, x_range, y_range, "Exponential"
