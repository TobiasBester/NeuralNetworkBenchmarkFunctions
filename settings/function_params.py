import numpy as np

import settings.function_definitions


def ackley():
    x_range = (-32, 32)
    y_range = x_range
    func = settings.function_definitions.ackley_func
    return func, x_range, y_range, "Ackley"


def ackley_n2():
    x_range = (-32, 32)
    y_range = x_range
    func = settings.function_definitions.ackley_n2_func
    return func, x_range, y_range, "Ackley N.2"


def ackley_n3():
    x_range = (-32, 32)
    y_range = x_range
    func = settings.function_definitions.ackley_n3_func
    return func, x_range, y_range, "Ackley N.3"


def ackley_n4():
    x_range = (-35, 35)
    y_range = x_range
    func = settings.function_definitions.ackley_n4_func
    return func, x_range, y_range, "Ackley N.4"


def adjiman():
    x_range = (-1, 2)
    y_range = (-1, 1)
    func = settings.function_definitions.adjiman_func
    return func, x_range, y_range, "Adjiman"


def alpine_n1():
    x_range = (0, 10)
    y_range = x_range
    func = settings.function_definitions.alpine_n1_func
    return func, x_range, y_range, "Alpine N.1"


def alpine_n2():
    x_range = (0, 10)
    y_range = x_range
    func = settings.function_definitions.alpine_n2_func
    return func, x_range, y_range, "Alpine N.2"


def beale():
    x_range = (-4.5, 4.5)
    y_range = x_range
    func = settings.function_definitions.beale_func
    return func, x_range, y_range, "Beale"


def bartels_conn():
    x_range = (-500, 500)
    y_range = x_range
    func = settings.function_definitions.bartels_conn_func
    return func, x_range, y_range, "Bartels Conn"


def bird():
    x_range = (-2 * np.pi, 2 * np.pi)
    y_range = x_range
    func = settings.function_definitions.bird
    return func, x_range, y_range, "Bird"


def boha_n1():
    x_range = (-100, 100)
    y_range = x_range
    func = settings.function_definitions.bohachevsky_n1
    return func, x_range, y_range, "Bohachevsky N.1"


def boha_n2():
    x_range = (-100, 100)
    y_range = x_range
    func = settings.function_definitions.bohachevsky_n2
    return func, x_range, y_range, "Bohachevsky N.2"


def booth():
    x_range = (-10, 10)
    y_range = x_range
    func = settings.function_definitions.booth
    return func, x_range, y_range, "Booth"


def brent():
    x_range = (-20, 0)
    y_range = x_range
    func = settings.function_definitions.brent
    return func, x_range, y_range, "Brent"


def brown():
    x_range = (-1, 1)
    y_range = x_range
    func = settings.function_definitions.brown
    return func, x_range, y_range, "Brown"


def bukin_n6():
    x_range = (-15, -5)
    y_range = (-3, 3)
    func = settings.function_definitions.bukin_n6
    return func, x_range, y_range, "Bukin N.6"


def cross_in_tray():
    x_range = (-10, 10)
    y_range = x_range
    func = settings.function_definitions.cross_in_tray
    return func, x_range, y_range, "Cross-In-Tray"


def deckkers_arts():
    x_range = (-20, 20)
    y_range = x_range
    func = settings.function_definitions.deckkers_aarts
    return func, x_range, y_range, "Deckkers-Aarts"


def drop_wave():
    x_range = (-5.2, 5.2)
    y_range = x_range
    func = settings.function_definitions.drop_wave
    return func, x_range, y_range, "Drop-Wave"


def easom():
    x_range = (-50, 50)
    y_range = x_range
    func = settings.function_definitions.easom
    return func, x_range, y_range, "Easom"


def egg_crate():
    x_range = (-5, 5)
    y_range = x_range
    func = settings.function_definitions.egg_crate
    return func, x_range, y_range, "Egg Crate"


def egg_holder():
    x_range = (-512, 512)
    y_range = x_range
    func = settings.function_definitions.egg_holder
    return func, x_range, y_range, "Egg Holder"


def exponential():
    x_range = (-1, 1)
    y_range = x_range
    func = settings.function_definitions.exponential
    return func, x_range, y_range, "Exponential"


def goldstein_price():
    x_range = (-2, 2)
    y_range = x_range
    func = settings.function_definitions.goldstein_price
    return func, x_range, y_range, "Goldstein-Price"


def griewank():
    x_range = (-600, 600)
    y_range = x_range
    func = settings.function_definitions.griewank
    return func, x_range, y_range, "Griewank"


def himmelblau():
    x_range = (-6, 6)
    y_range = x_range
    func = settings.function_definitions.himmelblau
    return func, x_range, y_range, "Himmelblau"


def holder_table():
    x_range = (-10, 10)
    y_range = x_range
    func = settings.function_definitions.holder_table
    return func, x_range, y_range, "Holder-Table"


def keane():
    x_range = (0, 10)
    y_range = x_range
    func = settings.function_definitions.keane
    return func, x_range, y_range, "Keane"


def leon():
    x_range = (0, 10)
    y_range = x_range
    func = settings.function_definitions.leon
    return func, x_range, y_range, "Leon"


def levi_n13():
    x_range = (-10, 10)
    y_range = x_range
    func = settings.function_definitions.levi_n13
    return func, x_range, y_range, "Levi N.13"


def matyas():
    x_range = (-10, 10)
    y_range = x_range
    func = settings.function_definitions.matyas
    return func, x_range, y_range, "Matyas"


def mccormick():
    x_range = (-1.5, 4)
    y_range = (-3, 3)
    func = settings.function_definitions.mccormick
    return func, x_range, y_range, "McCormick"
