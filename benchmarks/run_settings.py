import benchmarks.functions
import tensorflow as tf


def beale():
    x_range = (-4.5, 4.5)
    y_range = x_range
    func = benchmarks.functions.beale_func
    return func, x_range, y_range, "Beale"


def ackley_n2():
    x_range = (-32, 32)
    y_range = x_range
    func = benchmarks.functions.ackley_n2_func
    return func, x_range, y_range, "Ackley N.2"


def adjiman():
    x_range = (-1, 2)
    y_range = (-1, 1)
    func = benchmarks.functions.adjiman_func
    return func, x_range, y_range, "Adjiman"


def sgd_optimizer(learning_rate=0.05):
    return tf.optimizers.SGD(learning_rate), learning_rate, "SGD"


def adam_optimizer(learning_rate=0.001):
    return tf.optimizers.Adam(learning_rate), learning_rate, "Adam"
