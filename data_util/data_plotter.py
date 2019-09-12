from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from mpl_toolkits.mplot3d import Axes3D

from data_util.data_generator import normalize_data
from data_util.data_saver import check_directory


def plot_true_function(x_range, y_range, benchmark_func):
    x_step = (np.abs(x_range[0] - x_range[1])) / 2500
    x = np.arange(x_range[0], x_range[1], x_step)

    y_step = (np.abs(y_range[0] - y_range[1])) / 2500
    y = np.arange(y_range[0], y_range[1], y_step)

    z = benchmark_func(x, y)

    plot_2d_graph(x, z, 'True function in 2D')
    plt.show()
    plot_3d_graph(x, y, benchmark_func, 'True function in 3D')


def plot_2d_graph(x, z, title='', color='blue'):
    plt.scatter(x, z, color=color)
    plt.xlabel('x')
    plt.ylabel('f(x, y)')
    plt.title(title)


def plot_3d_graph(x, y, benchmark_func, title):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    x, y = np.meshgrid(x, y)
    func = benchmark_func(x, y)

    surface = ax.plot_surface(x, y, func, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    fig.colorbar(surface, shrink=0.5, aspect=5)

    plt.title(title)

    plt.show()


def plot_2d_predicted_vs_true(train_dataset, train_predictions, function_def):
    generated_x = [a[0] for a in train_dataset]
    generated_y = [a[1] for a in train_dataset]
    true_fitness = np.array([function_def(x, y) for (x, y) in train_dataset])
    normed_true_fitness = normalize_data(true_fitness)

    plot_2d_graph(generated_x, train_predictions, color='red')
    plot_2d_graph(generated_x, normed_true_fitness, 'Predicted f(x) (R) vs Actual f(x) (G)', 'green')
    plt.show()

    plot_2d_graph(generated_y, train_predictions, color='red')
    plot_2d_graph(generated_y, normed_true_fitness, 'Predicted f(y) (R) vs Actual f(y) (G)', 'green')
    plt.show()


def save_nn_results_to_file(nn_params, data_params, train_mse, test_mse):
    results_path = './results/nn_results'
    check_directory(results_path)

    file_name = "%s/%s__nn_results.txt" % (results_path, data_params.function_name)

    f = open(file_name, "a+")
    f.write(
        "===== New Run at %s =====\n"
        "Function: %s\n"
        "Num epochs: %d\n"
        "Optimizer: %s\n"
        "Num. hidden neurons: %d\n"
        "- Results -\n"
        "Train MSE: %.5f\n"
        "Test MSE: %.8f\n" %
        (
            datetime.now(),
            data_params.function_name,
            nn_params.num_epochs,
            nn_params.optimizer_name,
            nn_params.hidden_neurons,
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

    plt.plot(nn_train_history['mse'], color='green', label='NN Training MSE')
    plt.plot(nn_train_history['val_mse'], color='navy', label='NN Validation MSE')

    plt.axhline(y=lr_train_mse, color='indianred', label='LR Train MSE')
    plt.axhline(y=lr_test_mse, color='goldenrod', label='LR Test MSE')

    plt.legend()
    plt.show()
