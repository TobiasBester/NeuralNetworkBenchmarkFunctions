from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from mpl_toolkits.mplot3d import Axes3D

from data_util import plot_colors
from data_util.data_saver import check_directory


def plot_true_function(x_range, y_range, benchmark_func, func_name):
    x_step = (np.abs(x_range[0] - x_range[1])) / 2500
    x = np.arange(x_range[0], x_range[1], x_step)

    y_step = (np.abs(y_range[0] - y_range[1])) / 2500
    y = np.arange(y_range[0], y_range[1], y_step)

    z = benchmark_func(x, y)

    path = './results/true_functions/'
    check_directory(path)
    save_plot_to = "{}{}.png".format(path, func_name)

    plot_3d_graph(x, y, benchmark_func, 'True function in 3D', save_plot_to)


def plot_2d_graph(x, z, title='', color=plot_colors.BLUE):
    plt.scatter(x, z, color=color)
    plt.xlabel('x')
    plt.ylabel('f(x, y)')
    plt.title(title)


def plot_3d_graph(x, y, benchmark_func, title, dest=''):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    x, y = np.meshgrid(x, y)
    func = benchmark_func(x, y)

    surface = ax.plot_surface(x, y, func, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    fig.colorbar(surface, shrink=0.5, aspect=5)

    plt.title(title)

    if dest:
        plt.savefig(dest, format='png')

    plt.show()


def plot_2d_predicted_vs_true(x, y, predictions, true):
    plot_2d_graph(x, predictions, color=plot_colors.BLUE)
    plot_2d_graph(x, true, 'Predicted f(x) (Red) vs Actual f(x) (Blue)', plot_colors.RED)
    plt.show()

    plot_2d_graph(y, predictions, color=plot_colors.BLUE)
    plot_2d_graph(y, true, 'Predicted f(y) (Red) vs Actual f(y) (Blue)', plot_colors.RED)
    plt.show()


def plot_3d_predicted_vs_true(x, y, predictions, true, title='Predicted function (Blue) vs True function (Purple)',
                              dest=''):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x, y, predictions, color=plot_colors.BLUE)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')

    ax.scatter(x, y, true, color=plot_colors.PURPLE, marker='^')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')

    plt.title(title)

    if dest:
        plt.savefig(dest, format='png')

    plt.show()


def save_nn_results_to_file(nn_params, data_params, train_mse, test_mse, num_epochs):
    results_path = './results/nn_results'
    check_directory(results_path)

    file_name = "%s/%s__nn_results.txt" % (results_path, data_params.function_name)

    f = open(file_name, "a+")
    f.write(
        "===== New Run at %s =====\n"
        "Function: %s\n"
        "Max epochs: %d\n"
        "Optimizer: %s\n"
        "Num. hidden neurons: %d\n"
        "Num. epochs run: %d\n"
        "- Results -\n"
        "Train MSE: %.5f\n"
        "Test MSE: %.8f\n" %
        (
            datetime.now(),
            data_params.function_name,
            nn_params.num_epochs,
            nn_params.optimizer_name,
            nn_params.hidden_neurons,
            num_epochs,
            train_mse,
            test_mse)
    )
    f.close()

    print('Results saved to', file_name)


def save_lr_results_to_file(data_params, train_mse, test_mse):
    results_path = './results/lr_results'
    check_directory(results_path)

    file_name = "%s/%s__lr_results.txt" % (results_path, data_params.function_name)

    f = open(file_name, "a+")

    f.write(
        "===== New Run at %s =====\n"
        "Function: %s\n"
        "- Results -\n"
        "Train MSE: %.5f\n"
        "Test MSE: %.8f\n" %
        (
            datetime.now(),
            data_params.function_name,
            train_mse,
            test_mse)
    )
    f.close()

    print('Results saved to', file_name)


def plot_nn_and_lr_mse(lr_train_mse, lr_test_mse, nn_train_history):
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')

    plt.plot(nn_train_history['mse'], color=plot_colors.PURPLE, label='NN Training MSE')
    plt.plot(nn_train_history['val_mse'], color=plot_colors.BLUE, label='NN Validation MSE')

    plt.axhline(y=lr_train_mse, color=plot_colors.RED, label='LR Train MSE')
    plt.axhline(y=lr_test_mse, color=plot_colors.CREAM, label='LR Test MSE')

    plt.legend()
    plt.show()
