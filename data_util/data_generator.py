import numpy as np
import pandas as pd

# noinspection PyCompatibility
from configparser import ConfigParser

from settings.dataset_parameters import DatasetParameters


def data_setup():
    datagen_config_path = './config/datagen_config.ini'
    datagen_config = ConfigParser()
    datagen_config.read(datagen_config_path)

    section = 'DEFAULT'

    # np.random.seed(datagen_config.getint(section, 'random_seed'))
    # tf.random.set_seed(datagen_config.getint(section, 'random_seed'))

    return DatasetParameters(
        datagen_config.getint(section, 'datagen_seed'),
        datagen_config.getint(section, 'num_samples'),
        datagen_config.getboolean(section, 'show_true_function'),
        datagen_config.getboolean(section, 'show_generated_data'),
        datagen_config.getboolean(section, 'save_generated_data'),
        datagen_config.getboolean(section, 'show_predicted_vs_true'),
        datagen_config.get(section, 'function')
    )


def generate_random_dataset(x_range, y_range, num_samples, benchmark_func, function_name):
    x = np.random.uniform(x_range[0], x_range[1], num_samples)
    y = np.random.uniform(y_range[0], y_range[1], num_samples)
    func = benchmark_func(x, y)

    data = {'x': x, 'y': y, 'func': func}

    return pd.DataFrame().from_dict(data)


def generate_regularly_spaced_dataset(x_range, y_range, num_samples, benchmark_func):
    x = np.linspace(start=x_range[0], stop=x_range[1], num=num_samples)
    y = np.linspace(start=y_range[0], stop=y_range[1], num=num_samples)
    func = benchmark_func(x, y)

    data = {'x': x, 'y': y, 'func': func}

    return pd.DataFrame().from_dict(data)