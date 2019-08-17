from os import path, makedirs
from datetime import datetime

from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt


def fit_model(model, train_dataset, train_labels, num_epochs, show_history):
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
    history = model.fit(train_dataset,
                        train_labels,
                        epochs=num_epochs,
                        validation_split=0.2,
                        verbose=show_history,
                        callbacks=[early_stop])

    if show_history:
        print('Plotting history')
        plot_history(history)

    return history.history


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$Func^2$]')
    plt.plot(hist['epoch'], hist['mse'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'], label='Validation Error')
    plt.legend()

    plt.show()

    print('Final training loss, mae, mse:',
          history.history['loss'][-1],
          history.history['mae'][-1],
          history.history['mse'][-1])
    print('Final validation loss, mae, mse:',
          history.history['val_loss'][-1],
          history.history['val_mae'][-1],
          history.history['val_mse'][-1])


def try_out_model(model, train_data):
    example_batch = train_data[:10]
    example_result = model.predict(example_batch)
    print(example_result)


def evaluate_model(model, test_dataset, test_labels, nn_params, verbose):
    loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose)

    print('Testing set loss, mae, mse:', loss, mae, mse)

    return mse


def save_results_to_file(nn_params, train_history, test_mse):
    results_path = '../results'
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
        "Test MSE: %.5f\n" %
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


def check_directory(results_path):
    if not path.exists(results_path):
        makedirs(results_path)
