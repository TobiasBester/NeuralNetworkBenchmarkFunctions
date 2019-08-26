from __future__ import absolute_import, division, print_function

from LinearRegression import run_lr
from NeuralNetwork import run_nn
from data_util.data_generator import generate_random_dataset
from data_util.data_plotter import plot_true_function, plot_nn_and_lr_mse
from data_util.data_setup import data_setup, nn_setup
from data_util.data_splitter import split_data_for_lr, split_data_for_nn, save_generated_nn_data_to_file


def compare():
    print('== Setting up Data ==')
    data_params = data_setup()

    print('== Setting up Neural Network ==')
    nn_params = nn_setup()

    if data_params.show_true_function:
        plot_true_function(data_params.x_range, data_params.y_range, data_params.function_definition)

    print('== Generating data ==')
    dataset = generate_random_dataset(
        data_params.x_range,
        data_params.y_range,
        data_params.num_samples,
        data_params.function_definition,
        data_params.function_name
    )

    print('== Splitting data ==')
    lr_dataset_group = split_data_for_lr(
        dataset,
        data_params.function_definition,
        data_params.show_generated_data)

    nn_dataset_group = split_data_for_nn(
        dataset,
        data_params.function_definition,
        data_params.show_generated_data)

    if data_params.save_generated_data:
        save_generated_nn_data_to_file(data_params.function_name, nn_dataset_group)

    lr_train_mse, lr_test_mse = run_lr(data_params, lr_dataset_group)
    nn_test_mse, nn_train_history = run_nn(data_params, nn_params, nn_dataset_group)

    plot_nn_and_lr_mse(lr_train_mse, lr_test_mse, nn_train_history)


compare()
