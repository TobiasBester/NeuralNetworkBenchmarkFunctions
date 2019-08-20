from os import path, makedirs
from datetime import datetime

import matplotlib.pyplot as plt


def plot_results(dataset_group, test_predictions):
    test_x = [x[0] for x in dataset_group.test_dataset]
    test_y = [x[1] for x in dataset_group.test_dataset]

    plt.scatter(test_x, dataset_group.test_labels, color='blue')
    plt.xlabel('test x')
    plt.ylabel('test func')
    plt.show()

    plt.scatter(test_y, dataset_group.test_labels, color='red')
    plt.xlabel('test y')
    plt.ylabel('test func')
    plt.show()

    plt.scatter(test_x, test_predictions, color='blue')
    plt.xlabel('test x')
    plt.ylabel('predicted test func')
    plt.show()

    plt.scatter(test_y, test_predictions, color='red')
    plt.xlabel('test y')
    plt.ylabel('predicted test func')
    plt.show()


def save_nn_results_to_file(nn_params, train_history, test_mse):
    results_path = './results'
    check_directory(results_path)

    file_name = "%s/%s__nn_results.txt" % (results_path, nn_params.function_name)

    f = open(file_name, "a+")
    f.write(
        "===== New Run at %s =====\n"
        "Function: %s\n"
        "Num epochs: %d\n"
        "Optimizer: %s\n"
        "Num. hidden neurons: %d\n"
        "- Results -\n"
        "Train MSE: %.5f\n"
        "Validation MSE: %.5f\n"
        "Test MSE: %.8f\n" %
        (
            datetime.now(),
            nn_params.function_name,
            nn_params.num_epochs,
            nn_params.optimizer_name,
            nn_params.hidden_neurons,
            train_history['mse'][-1],
            train_history['val_mse'][-1],
            test_mse)
    )
    f.close()

    print('Results saved to', file_name)


def save_lr_results_to_file(lr_params, train_mse, test_mse):
    results_path = './results'
    check_directory(results_path)

    file_name = "%s/%s__lr_results.txt" % (results_path, lr_params.function_name)

    f = open(file_name, "a+")

    f.write(
        "===== New Run at %s =====\n"
        "Function: %s\n"
        "- Results -\n"
        "Train MSE: %.5f\n"
        "Test MSE: %.8f\n" %
        (
            datetime.now(),
            lr_params.function_name,
            train_mse,
            test_mse)
    )
    f.close()

    print('Results saved to', file_name)


def check_directory(results_path):
    if not path.exists(results_path):
        makedirs(results_path)
