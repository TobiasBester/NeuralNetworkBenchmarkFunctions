from configparser import ConfigParser

from sklearn import linear_model
from sklearn.metrics import mean_squared_error

import numpy as np

from data_util.data_generator import generate_random_dataset
from data_util.data_splitter import split_data_for_lr
from data_util.data_plotter import plot_results, save_lr_results_to_file
from settings.linear_regression_parameters import LinearRegressionParameters


def setup():
    config_path = './config/lr_config.ini'
    config = ConfigParser()
    config.read(config_path)

    section = 'DEFAULT'

    np.random.seed(config.getint(section, 'random_seed'))

    return LinearRegressionParameters(
        config.getint(section, 'num_samples'),
        config.get(section, 'function')
    ), config.getboolean(section, 'verbose')


def main():

    print('1. Setting up Linear Regression')
    lr_params, verbose = setup()
    print('Performing Linear Regression for', lr_params.function_name)

    print('2. Generating data')
    dataset = generate_random_dataset(
        lr_params.x_range,
        lr_params.y_range,
        lr_params.num_samples,
        lr_params.function_definition
    )

    print('3. Splitting and organizing data')
    dataset_group = split_data_for_lr(dataset)

    print('4. Creating Linear Regression Model')
    regr = linear_model.LinearRegression()

    print('5. Fitting training data to model')
    regr.fit(dataset_group.train_dataset, dataset_group.train_labels)

    print('6. Making predictions on training and test data')
    train_predictions = regr.predict(dataset_group.train_dataset)
    test_predictions = regr.predict(dataset_group.test_dataset)

    train_mse = mean_squared_error(y_true=dataset_group.train_labels, y_pred=train_predictions)
    test_mse = mean_squared_error(y_true=dataset_group.test_labels, y_pred=test_predictions)
    print("Train MSE:", train_mse)
    print("Test MSE:", test_mse)

    if verbose:
        plot_results(dataset_group, test_predictions)

    print('7. Saving results to file')
    save_lr_results_to_file(lr_params, train_mse, test_mse)


main()
