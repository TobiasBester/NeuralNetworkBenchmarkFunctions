from __future__ import absolute_import, division, print_function

import numpy as np

from pipelines.linear_regression_pipeline import run_lr
from pipelines.neural_network_pipeline import run_nn
from data_util.data_generator import generate_random_dataset
from data_util.data_plotter import plot_true_function, plot_nn_and_lr_mse
from data_util.data_saver import save_generated_nn_data_to_file, save_combined_results_to_file
from data_util.data_setup import data_setup, nn_setup
from data_util.data_splitter import split_data_for_lr, split_data_for_nn

if __name__ == '__main__':
    print('COMPARING METHODS')

    # warmup_gpu()

    data_params = data_setup()

    print('== Objective function:', data_params.function_name)

    print('== Setting up Neural Network ==')
    nn_params = nn_setup()

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

    test_mse_history = []
    num_epochs_history = []
    for idx, seed in enumerate(range(nn_params.starting_seed, nn_params.starting_seed + nn_params.num_seeds)):
        print('\nNN', idx)
        nn_test_mse, nn_train_history, num_epochs_run = run_nn(data_params, nn_params, nn_dataset_group, seed)

        if data_params.show_predicted_vs_true:
            plot_nn_and_lr_mse(lr_train_mse, lr_test_mse, nn_train_history)
        test_mse_history.append(nn_test_mse)
        num_epochs_history.append(num_epochs_run)

    average_nn_test_mse = np.mean(np.array(test_mse_history))
    stdev_nn_test_mse = np.std(np.array(test_mse_history))
    average_nn_epochs_run = np.mean(np.array(num_epochs_history))

    mse_index = lr_test_mse / average_nn_test_mse - 1

    save_combined_results_to_file(
        data_params.function_name,
        average_nn_epochs_run,
        average_nn_test_mse,
        stdev_nn_test_mse,
        lr_test_mse,
        mse_index
    )

    print('\nTest MSEs: LR = {} vs NN = {} ~ {}'.format(lr_test_mse, average_nn_test_mse, stdev_nn_test_mse))
    print('MSE Index:', mse_index)
    print('LR performed better' if mse_index < 0 else 'NN performed better')

    print('\nComparer Program Completed')
