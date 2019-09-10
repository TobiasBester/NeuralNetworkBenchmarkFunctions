from sklearn import linear_model
from sklearn.metrics import mean_squared_error

from data_util.data_plotter import save_lr_results_to_file, plot_2d_predicted_vs_true


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
        plot_2d_predicted_vs_true(dataset_group.train_dataset, train_predictions, data_params.function_definition)

    print('== Saving LR results to file ==')
    save_lr_results_to_file(data_params, train_mse, test_mse)

    return train_mse, test_mse