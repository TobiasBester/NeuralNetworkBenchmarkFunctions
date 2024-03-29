import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

from data_util.data_generator import normalize_data
from data_util.data_plotter import save_lr_results_to_file, plot_2d_predicted_vs_true, plot_3d_predicted_vs_true
from data_util.data_saver import check_directory


def run_lr(
        data_params,
        dataset_group
):
    print('Undergoing Neural Network Training Process for', data_params.function_name)

    print('== Creating Linear Regression Model ==')
    regr = linear_model.LinearRegression()

    print('== Fitting training data to LR model ==')
    regr.fit(dataset_group.train_dataset, dataset_group.train_labels)

    print('== Making predictions on training and test data ==')
    train_predictions = regr.predict(dataset_group.train_dataset)
    test_predictions = regr.predict(dataset_group.test_dataset)

    train_mse = mean_squared_error(y_true=dataset_group.train_labels, y_pred=train_predictions)
    test_mse = mean_squared_error(y_true=dataset_group.test_labels, y_pred=test_predictions)
    print("Train MSE:", train_mse)
    print("TEST MSE:", test_mse)

    if data_params.show_predicted_vs_true:
        generated_x = [a[0] for a in dataset_group.test_dataset]
        generated_y = [a[1] for a in dataset_group.test_dataset]
        true_fitness = np.array([data_params.function_definition(x, y) for (x, y) in dataset_group.test_dataset])
        normed_true_fitness = normalize_data(true_fitness)

        path = './results/lr_plots/'
        check_directory(path)
        save_plot_to = "{}{}.png".format(path, data_params.function_name)

        plot_2d_predicted_vs_true(generated_x, generated_y, test_predictions, normed_true_fitness)
        plot_3d_predicted_vs_true(generated_x, generated_y, test_predictions, normed_true_fitness,
                                  'Test Data: LR Preds(R) vs True(G)', save_plot_to)

    print('== Saving LR results to file ==')
    save_lr_results_to_file(data_params, train_mse, test_mse)

    return train_mse, test_mse
