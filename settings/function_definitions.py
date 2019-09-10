from numpy import cos, sin, pi, exp, sqrt, abs


def ackley_func(x, y):
    a = 20
    b = 0.2
    c = 2 * pi
    return -a * exp(-b * sqrt(0.5 * x ** 2 + y ** 2)) \
           - exp(0.5 * (cos(c * x) + cos(c * y))) + a + exp(1)


def ackley_n2_func(x, y):
    return -200 * exp(-0.2 * sqrt(x ** 2 + y ** 2))


def ackley_n3_func(x, y):
    return ackley_n2_func(x, y) + (5 * exp(cos(3 * x) + sin(3 * y)))


def ackley_n4_func(x, y):
    return exp(-0.2) * sqrt(x ** 2 + y ** 2) + 3 * (cos(2 * x) + sin(2 * y))


def adjiman_func(x, y):
    return cos(x) * sin(y) - (x / (y**2 + 1))


def alpine_n1_func(x, y):
    return alpine_n1_helper(x) + alpine_n1_helper(y)


def alpine_n1_helper(x):
    return abs(x * sin(x) + 0.1 * x)


def alpine_n2_func(x, y):
    return alpine_n2_helper(x) * alpine_n2_helper(y)


def alpine_n2_helper(x):
    return sqrt(x) * sin(x)


def beale_func(x, y):
    return (1.5 - x + (x * y)) ** 2 + (2.25 - x + (x * y ** 2)) ** 2 + (2.625 - x + (x * y ** 3)) ** 2


def bartels_conn_func(x, y):
    return abs(x ** 2 + y ** 2 + (x * y)) + abs(sin(x)) + abs(cos(y))


def bird(x, y):
    return (sin(x) * exp((1 - cos(y)) ** 2)) \
           + (cos(y) * exp((1 - sin(x)) ** 2)) \
           + (x - y) ** 2


def bohachevsky_n1(x, y):
    return (x ** 2) + (2 * y ** 2) - (0.3 * cos(3 * pi * x)) \
        - (0.4 * cos(4 * pi * y)) + 0.7


def bohachevsky_n2(x, y):
    return (x ** 2) + (2 * y ** 2) \
           - (0.3 * cos(3 * pi * x) * cos(4 * pi * y)) + 0.3


def booth(x, y):
    return (x + (2 * y) - 7) ** 2 + ((2 * x) + y - 5) ** 2


def brent(x, y):
    return (x + 10) ** 2 + (y + 10) ** 2 + (exp(-(x ** 2) - (y ** 2)))


def brown(x, y):
    return brown_helper(x, y) + brown_helper(y, x)


def brown_helper(x, y):
    return (x ** 2) ** ((y ** 2) + 1)


def bukin_n6(x, y):
    return 100 * sqrt(abs(y - 0.01 * x ** 2)) + 0.01 * abs(x + 10)


def cross_in_tray(x, y):
    return -0.0001 * (abs(sin(x) * sin(y) * exp(cit_helper(x, y))) + 1) ** 0.1


def cit_helper(x, y):
    return abs(100 - (sqrt(x ** 2 + y ** 2) / pi))


def deckkers_aarts(x, y):
    return 10 ** 5 * x ** 2 + y ** 2 - \
           (x ** 2 + y ** 2) ** 2 + \
           10 ** -5 * (x ** 2 + y ** 2) ** 4


def drop_wave(x, y):
    return - (1 + cos(12 * sqrt(x ** 2 + y ** 2))) / \
           (0.5 * (x ** 2 + y ** 2) + 2)


def easom(x, y):
    return -cos(x) * cos(y) * exp(-(x - pi) ** 2 - (y - pi) ** 2)


def egg_crate(x, y):
    return x ** 2 + y ** 2 + 25 * (sin(x) ** 2 + sin(y) ** 2)


def exponential(x, y):
    return -exp(-0.5 * (x ** 2 + y ** 2))


def goldstein_price(x, y):
    return (1 + (x + y + 1) ** 2 * (19 - (14 * x) + (3 * x ** 2) - 14 * y + (6 * x * y) + (3 * y ** 2))) \
           * (30 + ((2 * x - 3 * y) ** 2) * (18 - 32 * x + (12 * x ** 2) + 4 * y - (36 * x * y) + (27 * y ** 2)))


def griewank(x, y):
    return 1 + (griewank_sum(x) + griewank_sum(y)) - (griewank_product(x, 1) * griewank_product(y, 2))


def griewank_sum(x):
    return x ** 2 / 4000


def griewank_product(x, i):
    return cos(x / sqrt(i))


def himmelblau(x, y):
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2


def holder_table(x, y):
    return -abs(sin(x) * cos(y) * exp(abs(1 - (sqrt(x ** 2 + y ** 2) / pi))))
