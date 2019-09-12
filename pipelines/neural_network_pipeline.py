from data_util.data_plotter import save_nn_results_to_file
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

    print('== Evaluating the NN model on the test data ==')
    test_mse = evaluate_model(
        model,
        dataset_group.test_dataset,
        dataset_group.test_labels,
        nn_params.show_training)

    print('== Saving NN results to text file ==')
    save_nn_results_to_file(nn_params, data_params, train_history['mse'][-1], test_mse)

    print('TEST MSE:', test_mse)

    return test_mse, train_history
