from __future__ import absolute_import, division, print_function

import numpy as np

from pipelines.neural_network_pipeline import run_nn
from data_util.data_plotter import plot_true_function
from data_util.data_saver import save_generated_nn_data_to_file
from data_util.data_setup import nn_setup, data_setup
from data_util.data_splitter import split_data_for_nn

from data_util.data_generator import generate_random_dataset


def main():
    print('== Setting up Neural Network runner ==')
    data_params = data_setup()
    nn_params = nn_setup()

    print('== Objective Function:', data_params.function_name)

    if data_params.show_true_function:
        plot_true_function(data_params.x_range, data_params.y_range, data_params.function_definition,
                           data_params.function_name)

    print('== Generating data ==')
    dataset = generate_random_dataset(
        data_params.x_range,
        data_params.y_range,
        data_params.num_samples,
        data_params.function_definition
    )

    print('== Splitting data ==')
    dataset_group = split_data_for_nn(
        dataset,
        data_params.function_definition,
        data_params.show_generated_data)

    if data_params.save_generated_data:
        save_generated_nn_data_to_file(data_params.function_name, dataset_group)

    test_mse_history = []
    num_epochs_history = []

    for idx, seed in enumerate(range(nn_params.starting_seed, nn_params.starting_seed + nn_params.num_seeds)):
        print('\nNN', idx)
        test_mse, train_history, num_epochs_run = run_nn(data_params, nn_params, dataset_group, seed)

        test_mse_history.append(test_mse)
        num_epochs_history.append(num_epochs_run)

    print('\nAverage Test MSE:', np.mean(np.array(test_mse_history)))
    print('\nStdev Test MSE:', np.std(np.array(test_mse_history)))
    print('\nAverage Num Epochs Run:', np.mean(np.array(num_epochs_history)))


main()

# TODO: c) Look for more functions
