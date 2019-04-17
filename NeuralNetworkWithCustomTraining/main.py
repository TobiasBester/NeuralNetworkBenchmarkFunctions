from __future__ import absolute_import, division, print_function

import configparser

import benchmarks.run_settings as settings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.mplot3d import Axes3D
from tensorflow.python import keras
from tensorflow.python.keras import layers

num_epochs = 50
num_samples = 2500
learning_rate = 0.05
benchmark_func, x_range, y_range, func_name = settings.beale()
optimizer, learning_rate, optimizer_name = settings.sgd_optimizer(learning_rate)


def generate_random_dataset():
    x = np.random.uniform(x_range[0], x_range[1], num_samples)
    y = np.random.uniform(y_range[0], y_range[1], num_samples)
    func = benchmark_func(x, y)

    data = {'x': x, 'y': y, 'func': func}

    return pd.DataFrame().from_dict(data)


def generate_regularly_spaced_dataset():
    x = np.linspace(start=x_range[0], stop=x_range[1], num=num_samples)
    y = np.linspace(start=y_range[0], stop=y_range[1], num=num_samples)
    func = benchmark_func(x, y)

    data = {'x': x, 'y': y, 'func': func}

    return pd.DataFrame().from_dict(data)


def split_data(dataset, inspect_data=False):
    train_dataset = dataset.sample(frac=0.75, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    train_stats = train_dataset.describe()

    if inspect_data:
        sns.lmplot(data=train_dataset, x='x', y='func')
        # experiment.log_figure(figure_name='Training Data Scatter Plot', figure=plt)
        plt.show()

        # plot_3d_graph(train_dataset['x'], train_dataset['y'], train_dataset['func'])

        print(train_stats)

    return train_dataset, test_dataset, train_stats


def plot_3d_graph(x, y, func):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    x, y = np.meshgrid(x, y)
    func = benchmark_func(x, y)

    graph = ax.plot_surface(x, y, func, linewidth=1)
    # experiment.log_figure(figure_name="Surface Plot of Generated Data", figure=plt)

    plt.show()


def split_features_from_labels(train_dataset, test_dataset):
    train_labels = train_dataset.pop('func')
    test_labels = test_dataset.pop('func')
    return train_labels, test_labels


def build_model(train_dataset):
    model = keras.Sequential([
        layers.Dense(4, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(4, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])

    return model


def try_out_model(model, train_data):
    example_batch = train_data[:10]
    example_result = model.predict(example_batch)
    print(example_result)


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [Func]')
    plt.xlim(left=0, right=50)
    plt.ylim(top=15000, bottom=0)
    plt.plot(hist['epoch'], hist['mae'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'], label='Validation Error')
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$Func^2$]')
    plt.xlim(left=0, right=50)
    plt.ylim(top=1000000000, bottom=0)
    plt.plot(hist['epoch'], hist['mse'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'], label='Validation Error')
    plt.legend()

    # experiment.log_figure(figure_name="History of MAE", figure=plt)
    plt.show()

    # experiment.log_metrics(history.history)

    print('Final loss, mae, mse:',
          history.history['loss'][-1],
          history.history['mae'][-1],
          history.history['mse'][-1])
    print('Final validation loss, mae, mse:',
          history.history['val_loss'][-1],
          history.history['val_mae'][-1],
          history.history['val_mse'][-1])


def plot_predictions(test_labels, test_predictions):
    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True Value [func]')
    plt.ylabel('Predictions [func]')
    _ = plt.plot()
    plt.show()


def main():
    np.random.seed(400)
    tf.random.set_seed(400)

    hyper_params = {"num_epochs": params['num_epochs'],
                    "learning_rate": params['learning_rate'],
                    "function": 'Beale',
                    "range": '[-4.5, 4.5]',
                    "optimizer": 'SGD'}
    # experiment.log_parameters(hyper_params)

    # 1. get dataset
    print('Obtaining training data')
    # dataset = generate_regularly_spaced_dataset()
    dataset = generate_random_dataset()

    # 2. clean and split the data
    print('Splitting data')
    train_dataset, test_dataset, train_stats = split_data(dataset, True)
    train_labels, test_labels = split_features_from_labels(train_dataset, test_dataset)

    # 3. build the model
    print('Building the model')
    model = build_model(train_dataset)
    # print(model.summary())

    # 4. train the model
    print('Training the model')
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit(train_dataset,
                        train_labels,
                        epochs=num_epochs,
                        validation_split=0.2,
                        verbose=0,
                        callbacks=[early_stop])
    print('Plotting history')
    plot_history(history)

    # 5. evaluate the model
    print('Evaluating the model on the test data')
    loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=0)
    print('Testing set loss, mae, mse:', loss, mae, mse)

    f = open("./results/nn_results.txt", "a+")
    f.write(
        "===== New Run =====\n"
        "Function %s\n"
        "Num epochs %d\n"
        "Optimizer %s\n"
        "Learning rate %f\n"
        "- Results -\n"
        "Test MSE %.5f\n" %
        (func_name, num_epochs, optimizer_name, learning_rate, mse)
    )
    f.close()


configs = configparser.ConfigParser()
configs.read('./config/config.properties')
params = globals()
# experiment = Experiment(api_key=configs['DEFAULT']['COMET_API_KEY'],
#                         project_name="nn-regression-benchmarks",
#                         workspace="tobiasbester",
#                         disabled=True)
main()
