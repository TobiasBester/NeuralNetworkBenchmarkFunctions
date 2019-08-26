from configparser import ConfigParser
import numpy as np
import tensorflow as tf

from settings.dataset_parameters import DatasetParameters
from settings.neural_network_parameters import NeuralNetworkParameters


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


def data_setup():
    datagen_config_path = './config/datagen_config.ini'
    datagen_config = ConfigParser()
    datagen_config.read(datagen_config_path)

    section = 'DEFAULT'

    np.random.seed(datagen_config.getint(section, 'datagen_seed'))
    tf.random.set_seed(datagen_config.getint(section, 'datagen_seed'))

    return DatasetParameters(
        datagen_config.getint(section, 'datagen_seed'),
        datagen_config.getint(section, 'num_samples'),
        datagen_config.getboolean(section, 'show_true_function'),
        datagen_config.getboolean(section, 'show_generated_data'),
        datagen_config.getboolean(section, 'save_generated_data'),
        datagen_config.getboolean(section, 'show_predicted_vs_true'),
        datagen_config.get(section, 'function')
    )
