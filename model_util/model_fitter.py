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
