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


def michalewicz():
    x_range = (0, np.pi)
    y_range = x_range
    func = settings.function_definitions.michalewicz
    return func, x_range, y_range, "Michalewicz"


def periodic():
    x_range = (-10, 10)
    y_range = x_range
    func = settings.function_definitions.periodic
    return func, x_range, y_range, "Periodic"


def qing():
    x_range = (-500, 500)
    y_range = x_range
    func = settings.function_definitions.qing
    return func, x_range, y_range, "Qing"


def rastrigin():
    x_range = (-5.12, 5.12)
    y_range = x_range
    func = settings.function_definitions.rastrigin
    return func, x_range, y_range, "Rastrigin"


def ridge():
    x_range = (-5, 5)
    y_range = x_range
    func = settings.function_definitions.ridge
    return func, x_range, y_range, "Ridge"


def rosenbrock():
    x_range = (-5, 10)
    y_range = x_range
    func = settings.function_definitions.rosenbrock
    return func, x_range, y_range, "Rosenbrock"


def salomon():
    x_range = (-100, 100)
    y_range = x_range
    func = settings.function_definitions.salomon
    return func, x_range, y_range, "Salomon"


def schaffer_n2():
    x_range = (-100, 100)
    y_range = x_range
    func = settings.function_definitions.schaffer_n2
    return func, x_range, y_range, "Schaffer N.2"


def schaffer_n3():
    x_range = (-100, 100)
    y_range = x_range
    func = settings.function_definitions.schaffer_n3
    return func, x_range, y_range, "Schaffer N.3"


def schwefel_220():
    x_range = (-100, 100)
    y_range = x_range
    func = settings.function_definitions.schwefel_220
    return func, x_range, y_range, "Schwefel 2.20"


def schwefel_222():
    x_range = (-100, 100)
    y_range = x_range
    func = settings.function_definitions.schwefel_222
    return func, x_range, y_range, "Schwefel 2.22"


def schwefel_223():
    x_range = (-10, 10)
    y_range = x_range
    func = settings.function_definitions.schwefel_223
    return func, x_range, y_range, "Schwefel 2.23"


def shubert_3():
    x_range = (-10, 10)
    y_range = x_range
    func = settings.function_definitions.shubert_3
    return func, x_range, y_range, "Shubert 3"


def shubert():
    x_range = (-10, 10)
    y_range = x_range
    func = settings.function_definitions.shubert
    return func, x_range, y_range, "Shubert"


def sphere():
    x_range = (-5.12, 5.12)
    y_range = x_range
    func = settings.function_definitions.sphere
    return func, x_range, y_range, "Sphere"


def styblinski_tang():
    x_range = (-5, 5)
    y_range = x_range
    func = settings.function_definitions.styblinski_tang
    return func, x_range, y_range, "Styblinski-Tang"


def sum_squares():
    x_range = (-10, 10)
    y_range = x_range
    func = settings.function_definitions.sum_squares
    return func, x_range, y_range, "Sum Squares"


def three_hump_camel():
    x_range = (-5, 5)
    y_range = x_range
    func = settings.function_definitions.three_hump_camel
    return func, x_range, y_range, "Three-Hump Camel"


def xin_she_yang_n2():
    x_range = (-2 * np.pi, 2 * np.pi)
    y_range = x_range
    func = settings.function_definitions.xin_she_yang_n2
    return func, x_range, y_range, "Xin-She Yang N.2"


def xin_she_yang_n3():
    x_range = (-2 * np.pi, 2 * np.pi)
    y_range = x_range
    func = settings.function_definitions.xin_she_yang_n3
    return func, x_range, y_range, "Xin-She Yang N.3"


def xin_she_yang_n4():
    x_range = (-10, 10)
    y_range = x_range
    func = settings.function_definitions.xin_she_yang_n4
    return func, x_range, y_range, "Xin-She Yang N.4"


def zakharov():
    x_range = (-5, 10)
    y_range = x_range
    func = settings.function_definitions.zakharov
    return func, x_range, y_range, "Zakharov"
