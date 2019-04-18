from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import benchmarks.run_settings as settings


num_samples = 2500
benchmark_func, x_range, y_range, func_name = settings.adjiman()


def generate_random_dataset():
    x = np.random.uniform(x_range[0], x_range[1], num_samples)
    y = np.random.uniform(y_range[0], y_range[1], num_samples)
    func = benchmark_func(x, y)

    data = {'x': x, 'y': y, 'func': func}

    return pd.DataFrame().from_dict(data)


def split_data(dataset):
    train_dataset = dataset.sample(frac=0.75, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)
    return train_dataset, test_dataset


def plot_3d_graph(x, y, func):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    x, y = np.meshgrid(x, y)
    func = benchmark_func(x, y)

    graph = ax.plot_surface(x, y, func, linewidth=1)
    plt.show()


def main():
    print('Performing linear regression')
    np.random.seed(400)

    print('Generating data')
    dataset = generate_random_dataset()
    train_dataset, test_dataset = split_data(dataset)

    print('Organizing data')
    train_input = list(zip(train_dataset['x'], train_dataset['y']))
    train_labels = list(train_dataset['func'])

    test_input = list(zip(test_dataset['x'], test_dataset['y']))
    test_labels = list(test_dataset['func'])

    print('Fitting training data')
    regr = linear_model.LinearRegression()
    regr.fit(train_input, train_labels)

    print('Making predictions on test data')
    test_predictions = regr.predict(test_input)

    mse = mean_squared_error(y_true=test_labels, y_pred=test_predictions)

    print('STATS')
    test_x = [x[0] for x in test_input]
    test_y = [x[1] for x in test_input]
    plot_3d_graph(test_x, test_y, test_labels)
    plt.scatter(test_x, test_labels, color='blue')
    plt.plot(test_x, test_predictions, color='red')

    file_name = "../results/%s__lr_results.txt" % func_name
    f = open(file_name, "a+")
    f.write(
        "===== New Run =====\n"
        "Function %s\n"
        "- Results -\n"
        "Test MSE %.5f\n" %
        (func_name, mse)
    )
    f.close()

    plt.show()


main()
