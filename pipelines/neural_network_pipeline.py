from data_util.data_plotter import save_nn_results_to_file, plot_3d_predicted_vs_true
from data_util.data_saver import check_directory
from model_util.model_builder import build_model
from model_util.model_fitter import fit_model, evaluate_model

import numpy as np
import tensorflow as tf


def run_nn(
        data_params,
        nn_params,
        dataset_group,
        seed
):
    print('=== Undergoing Neural Network Training Process with seed', seed)

    np.random.seed(seed)
    tf.random.set_seed(seed)

    print('== Building the NN model ==')
    model = build_model(
        dataset_group.train_dataset,
        nn_params.optimizer,
        nn_params.hidden_neurons,
        show_summary=nn_params.show_model_details
    )

    # try_out_model(model, dataset_group.train_dataset)

    print('== Training the NN model ==')
    train_history = fit_model(
        model,
        dataset_group.train_dataset,
        dataset_group.train_labels,
        nn_params.num_epochs,
        show_history=nn_params.show_training
    )

    num_epochs_run = len(train_history['mse'])

    print('== Evaluating the NN model on the test data ==')
    test_mse = evaluate_model(
        model,
        dataset_group.test_dataset,
        dataset_group.test_labels,
        nn_params.show_training)

    print('== Saving NN results to text file ==')
    save_nn_results_to_file(nn_params, data_params, train_history['mse'][-1], test_mse, num_epochs_run)

    if data_params.show_predicted_vs_true:
        test_preds = model.predict(dataset_group.test_dataset)
        test_x = dataset_group.test_dataset['x'].to_numpy()
        test_y = dataset_group.test_dataset['y'].to_numpy()
        test_true = dataset_group.test_labels.to_numpy()

        path = './results/nn_plots/'
        check_directory(path)
        save_plot_to = "{}{}.png".format(path, data_params.function_name)

        plot_3d_predicted_vs_true(test_x, test_y, test_preds, test_true, 'Test Data: NN Preds(R) vs True(G)',
                                  save_plot_to)

    print('TEST MSE:', test_mse)

    return test_mse, train_history, num_epochs_run
