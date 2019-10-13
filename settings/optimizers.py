import tensorflow as tf


def sgd_optimizer(learning_rate=0.01):
    return tf.optimizers.SGD(learning_rate), learning_rate, "SGD"


def adam_optimizer(learning_rate=0.001):
    return tf.optimizers.Adam(learning_rate), learning_rate, "Adam"
