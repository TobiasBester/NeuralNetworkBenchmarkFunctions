from __future__ import absolute_import, division, print_function

from pipelines.linear_regression_pipeline import run_lr
from data_util.data_generator import generate_random_dataset
from data_util.data_plotter import plot_true_function
from data_util.data_saver import save_generated_lr_data_to_file
from data_util.data_setup import data_setup
from data_util.data_splitter import split_data_for_lr


def main():
    print('== Setting up Linear Regression runner')
    data_params = data_setup()

    print('== Objective Function:', data_params.function_name)

    if data_params.show_true_function:
        plot_true_function(data_params.x_range, data_params.y_range, data_params.function_definition)

    print('== Generating data')
    dataset = generate_random_dataset(
        data_params.x_range,
        data_params.y_range,
        data_params.num_samples,
        data_params.function_definition
    )

    print('== Splitting data')
    dataset_group = split_data_for_lr(
        dataset,
        data_params.function_definition,
        data_params.show_generated_data)

    if data_params.save_generated_data:
        save_generated_lr_data_to_file(data_params.function_name, dataset_group)

    train_mse, test_mse = run_lr(data_params, dataset_group)


main()
