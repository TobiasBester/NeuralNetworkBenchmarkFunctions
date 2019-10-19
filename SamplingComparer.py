import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from data_util import plot_colors
from data_util.data_generator import generate_random_dataset, generate_uniform_random_dataset
from data_util.data_plotter import plot_2d_graph
from settings.dataset_parameters import parse_function

if __name__ == '__main__':
    np.random.seed(10)
    num_samples = 5000

    function_definition, x_range, y_range, function_name = parse_function('mccormick')
    uniform_dataset = generate_uniform_random_dataset(x_range, y_range, num_samples, function_definition)
    plot_2d_graph(uniform_dataset['x'], uniform_dataset['y'], title='Uniform sampling')
    plt.show()

    lhs_dataset = generate_random_dataset(x_range, y_range, num_samples, function_definition, 10)
    plot_2d_graph(lhs_dataset['x'], lhs_dataset['y'], title='LHS sampling')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(uniform_dataset['x'], uniform_dataset['y'], uniform_dataset['func'], color=plot_colors.BLUE)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    plt.title('Uniform Sampling')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(lhs_dataset['x'], lhs_dataset['y'], lhs_dataset['func'], color=plot_colors.RED)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    plt.title('LHS Sampling')
    plt.show()
