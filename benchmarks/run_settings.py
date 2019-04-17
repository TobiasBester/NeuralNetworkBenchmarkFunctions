import benchmarks.functions
import tensorflow as tf


def beale():
    x_range = (-4.5, 4.5)
    y_range = (-4.5, 4.5)
    func = benchmarks.functions.beale_func
    return func, x_range, y_range, "Beale"


def sgd_optimizer(learning_rate=0.05):
    return tf.optimizers.SGD(learning_rate), learning_rate, "SGD"


def adam_optimizer(learning_rate=0.001):
    return tf.optimizers.Adam(learning_rate), learning_rate, "Adam"
