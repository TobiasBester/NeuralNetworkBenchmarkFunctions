import settings.functions


def ackley_n2():
    x_range = (-32, 32)
    y_range = x_range
    func = settings.functions.ackley_n2_func
    return func, x_range, y_range, "Ackley N.2"


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
