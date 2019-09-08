from __future__ import absolute_import, division, print_function

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

    if data_params.show_true_function:
        plot_true_function(data_params.x_range, data_params.y_range, data_params.function_definition)

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

    test_mse, train_history = run_nn(data_params, nn_params, dataset_group)


main()

# TODO: a) Increase number of hidden neurons
# TODO: b) Investigate Linear Regression from scikit-learn
# TODO: b) Set different seeds for neural network runs (up to 10 and then use the Mean MSE)
# TODO: c) Look for more functions
# TODO: d) Deal with insufficient samples for functions like Easom
# TODO: d) With Latex producer, sort alphabetically and remove duplicate functions
# TODO: e) Plot true surface against NN predicted surface
