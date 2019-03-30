from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import tensorflow as tf

# Building and training a model involves a few steps
# 1. Define the model
# 2. Define a loss function
# 3. Obtain training data
# 4. Run through the training data and use an 'optimizer' to adjust the variables to fit the data
# In this tutorial, we'll work with the simple linear model f(x) = x * W + b
# We'll synthesize the data to be W = 3.0 and b = 2.0

TRUE_W = 3.0
TRUE_b = 2.0
NUM_EXAMPLES = 1000


class Model(object):                    # Step 1

    def __init__(self):
        # Initialize variable to (5.0, 0.0), although it should be random values
        self.W = tf.Variable(5.0)
        self.b = tf.Variable(0.0)

    def __call__(self, x):
        return self.W * x + self.b


def loss(predicted_y, desired_y):       # Step 2
    # Mean squared error
    return tf.reduce_mean(tf.square(predicted_y - desired_y))


def train(model, inputs, outputs, learning_rate):
    with tf.GradientTape() as t:
        current_loss = loss(model(inputs), outputs)
    dW, db = t.gradient(current_loss, [model.W, model.b])
    model.W.assign_sub(learning_rate * dW)
    model.b.assign_sub(learning_rate * db)


def main():

    # Step 3
    print('Obtaining training data')
    inputs = tf.random.normal(shape=[NUM_EXAMPLES])
    noise = tf.random.normal(shape=[NUM_EXAMPLES])
    outputs = inputs * TRUE_W + TRUE_b + noise

    model = Model()

    print('Visualizing model as of now')
    print('Training data is blue, Model predictions in red')
    plt.scatter(inputs, outputs, c='b')
    plt.scatter(inputs, model(inputs), c='r')
    plt.show()

    print('Current loss: '),
    print(loss(model(inputs), outputs).numpy())

    # Step 4
    print('Defining training loop')
    Ws, bs = [], []
    epochs = range(50)
    for epoch in epochs:
        Ws.append(model.W.numpy())
        bs.append(model.b.numpy())
        current_loss = loss(model(inputs), outputs)

        train(model, inputs, outputs, learning_rate=0.05)
        print('Epoch %2d: W=%1.2f b=%1.2f, loss=%2.5f' %
              (epoch, Ws[-1], bs[-1], current_loss))

    print('Plot the history of W and b')
    plt.plot(epochs, Ws, 'r',
             epochs, bs, 'b')
    plt.plot([TRUE_W] * len(epochs), 'r--',
             [TRUE_b] * len(epochs), 'b--')
    plt.legend(['W', 'b', 'true W', 'true b'])
    plt.show()


main()
