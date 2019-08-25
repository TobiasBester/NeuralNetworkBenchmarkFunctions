from os import path, makedirs
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

# from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import Axes3D


def plot_true_function(x_range, y_range, benchmark_func):
    x_step = (np.abs(x_range[0] - x_range[1])) / 2500
    x = np.arange(x_range[0], x_range[1], x_step)

    y_step = (np.abs(y_range[0] - y_range[1])) / 2500
    y = np.arange(y_range[0], y_range[1], y_step)

    z = benchmark_func(x, y)

    plot_2d_graph(x, z, 'True function in 2D')
    plot_3d_graph(x, y, benchmark_func, 'True function in 3D')


def plot_2d_graph(x, z, title):
    plt.scatter(x, z, color='blue')
    plt.xlabel('x')
    plt.ylabel('f(x, y)')
    plt.title(title)
    plt.show()


def plot_3d_graph(x, y, benchmark_func, title):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    x, y = np.meshgrid(x, y)
    func = benchmark_func(x, y)

    surface = ax.plot_surface(x, y, func, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    fig.colorbar(surface, shrink=0.5, aspect=5)

    plt.title(title)

    plt.show()


def plot_results(dataset_group, test_predictions):
    test_x = [x[0] for x in dataset_group.test_dataset]
    test_y = [x[1] for x in dataset_group.test_dataset]

    plt.scatter(test_x, dataset_group.test_labels, color='blue')
    plt.xlabel('test x')
    plt.ylabel('test func')
    plt.show()

    plt.scatter(test_y, dataset_group.test_labels, color='red')
    plt.xlabel('test y')
    plt.ylabel('test func')
    plt.show()

    plt.scatter(test_x, test_predictions, color='blue')
    plt.xlabel('test x')
    plt.ylabel('predicted test func')
    plt.show()

    plt.scatter(test_y, test_predictions, color='red')
    plt.xlabel('test y')
    plt.ylabel('predicted test func')
    plt.show()


def save_nn_results_to_file(nn_params, train_history, test_mse):
    results_path = './results'
    check_directory(results_path)

    file_name = "%s/%s__nn_results.txt" % (results_path, nn_params.function_name)

    f = open(file_name, "a+")
    f.write(
        "===== New Run at %s =====\n"
        "Function: %s\n"
        "Num epochs: %d\n"
        "Optimizer: %s\n"
        "Num. hidden neurons: %d\n"
        "- Results -\n"
        "Train MSE: %.5f\n"
        "Validation MSE: %.5f\n"
        "Test MSE: %.8f\n" %
        (
            datetime.now(),
            nn_params.function_name,
            nn_params.num_epochs,
            nn_params.optimizer_name,
            nn_params.hidden_neurons,
            train_history['mse'][-1],
            train_history['val_mse'][-1],
            test_mse)
    )
    f.close()

    print('Results saved to', file_name)


def save_lr_results_to_file(lr_params, train_mse, test_mse):
    results_path = './results'
    check_directory(results_path)

    file_name = "%s/%s__lr_results.txt" % (results_path, lr_params.function_name)

    f = open(file_name, "a+")

    f.write(
        "===== New Run at %s =====\n"
        "Function: %s\n"
        "- Results -\n"
        "Train MSE: %.5f\n"
        "Test MSE: %.8f\n" %
        (
            datetime.now(),
            lr_params.function_name,
            train_mse,
            test_mse)
    )
    f.close()

    print('Results saved to', file_name)


def check_directory(results_path):
    if not path.exists(results_path):
        makedirs(results_path)
