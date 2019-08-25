from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

# noinspection PyCompatibility
from configparser import ConfigParser

from NeuralNetwork import run
from data_util.data_plotter import plot_true_function
from data_util.data_splitter import split_data_for_nn, save_generated_nn_data_to_file
from settings.neural_network_parameters import NeuralNetworkParameters

from data_util.data_generator import data_setup, generate_random_dataset


def nn_setup():
    nn_config_path = './config/nn_config.ini'
    nn_config = ConfigParser()
    nn_config.read(nn_config_path)

    section = 'DEFAULT'

    return NeuralNetworkParameters(
        nn_config.getint(section, 'num_seeds_to_try'),
        nn_config.getint(section, 'starting_seed'),
        nn_config.getint(section, 'num_epochs'),
        nn_config.getint(section, 'neurons_in_hidden_layer'),
        nn_config.get(section, 'optimizer'),
        nn_config.getboolean(section, 'show_model_details'),
        nn_config.getboolean(section, 'show_training_process'),
    )


def main():
    print('1. Setting up Neural Network runner')
    data_params = data_setup()
    nn_params = nn_setup()

    if data_params.show_true_function:
        plot_true_function(
            data_params.x_range,
            data_params.y_range,
            data_params.function_definition)

    np.random.seed(data_params.seed)
    tf.random.set_seed(data_params.seed)

    print('2. Generating data')
    dataset = generate_random_dataset(
        data_params.x_range,
        data_params.y_range,
        data_params.num_samples,
        data_params.function_definition,
        data_params.function_name
    )

    print('3. Splitting data')
    dataset_group = split_data_for_nn(
        dataset,
        data_params.function_definition,
        data_params.show_generated_data)

    if data_params.save_generated_data:
        save_generated_nn_data_to_file(data_params.function_name, dataset_group)

    # run()


main()
