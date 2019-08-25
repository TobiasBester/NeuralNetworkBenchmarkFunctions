from sklearn import linear_model
from sklearn.metrics import mean_squared_error

from data_util.data_plotter import save_lr_results_to_file


def run(
        data_params,
        dataset_group
):

    print('Undergoing Neural Network Training Process for', data_params.function_name)

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
    print("TEST MSE:", test_mse)

    print('7. Saving results to file')
    save_lr_results_to_file(data_params, train_mse, test_mse)

    return test_mse
