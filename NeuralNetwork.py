from data_util.data_plotter import save_nn_results_to_file
from model_util.model_builder import build_model
from model_util.model_fitter import fit_model, evaluate_model


def run(
        data_params,
        nn_params,
        dataset
):

    # New
    # 1 Show x-z for uniform range v/
    # 2 Show x-y-z for uniform range v/
    # 4 Show data split stats
    # 3 Show x-y for random data
    # ---
    # LR
    # 10 Show LR MSE on same plot as 7
    # 7 Show train and val MSE over time
    # NN
    # 5 Show model details
    # 6 Show training process
    # 8 Eval model on test - show training
    # 9 Show predicted model on test data - plot it separately and over x-y-z

    print('Undergoing Neural Network Training Process for', nn_params.function_name)

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
