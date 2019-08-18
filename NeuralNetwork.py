from __future__ import absolute_import, division, print_function

# noinspection PyCompatibility
from configparser import ConfigParser

import numpy as np
import tensorflow as tf

from model_util.results_processor import save_nn_results_to_file
from model_util.model_builder import build_model
from model_util.model_fitter import fit_model, evaluate_model
from settings.neural_network_parameters import NeuralNetworkParameters
from data_util.data_generator import generate_random_dataset
from data_util.data_splitter import split_data_for_nn


def setup():
    config_path = './config/nn_config.ini'
    config = ConfigParser()
    config.read(config_path)

    section = 'DEFAULT'

    np.random.seed(config.getint(section, 'random_seed'))
    tf.random.set_seed(config.getint(section, 'random_seed'))

    return NeuralNetworkParameters(
        config.getint(section, 'num_epochs'),
        config.getint(section, 'num_samples'),
        config.get(section, 'function'),
        config.get(section, 'optimizer'),
        config.getint(section, 'neurons_in_hidden_layer')
    ), config.getboolean(section, 'verbose')


def main():

    print('1. Setting up Neural Network')
    nn_params, verbose = setup()
    print('Undergoing Neural Network Training Process for', nn_params.function_name)

    print('2. Generating data')
    dataset = generate_random_dataset(
        nn_params.x_range,
        nn_params.y_range,
        nn_params.num_samples,
        nn_params.function_definition
    )

    print('3. Splitting data')
    dataset_group = split_data_for_nn(dataset, benchmark_func=nn_params.function_definition, inspect_data=verbose)

    print('4. Building the model')
    model = build_model(
        dataset_group.train_dataset,
        nn_params.optimizer,
        nn_params.hidden_neurons,
        show_summary=verbose
    )

    # try_out_model(model, dataset_group.train_dataset)

    print('5. Training the model')
    train_history = fit_model(
        model,
        dataset_group.train_dataset,
        dataset_group.train_labels,
        nn_params.num_epochs,
        show_history=verbose
    )

    print('6. Evaluating the model on the test data')
    test_mse = evaluate_model(model, dataset_group.test_dataset, dataset_group.test_labels, nn_params, verbose)

    print('7. Saving results to text file')
    save_nn_results_to_file(nn_params, train_history, test_mse)


main()
