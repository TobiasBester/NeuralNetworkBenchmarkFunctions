from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers

EPOCHS = 1000


def get_dataset():
    # Get autoMPG dataset
    dataset_path = keras.utils.get_file("auto-mpg.data",
                                        "https://archive.ics.uci.edu/ml/machine-learning-databases/auto"
                                        "-mpg/auto-mpg.data")

    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                    'Acceleration', 'Model Year', 'Origin']
    # import the dataset to pandas
    raw_dataset = pd.read_csv(dataset_path,
                              names=column_names,
                              na_values='?',
                              comment='\t',
                              sep=' ',
                              skipinitialspace=True)
    dataset = raw_dataset.copy()
    print(dataset.tail())

    return dataset


def clean_data(dataset):
    # drop the rows with unknown values
    dataset = dataset.dropna()

    # convert the "Origin" column from categorical to numerical via one-hotting
    #   one-hotting is representing categorical variables as binary vectors
    #       this first requires that the categorical values be mapped to integer values
    #       then each integer value is represented as a binary vector that is all zeros except the index of the integer,
    #        which is a 1
    origin = dataset.pop('Origin')
    dataset['USA'] = (origin == 1) * 1.0
    dataset['Europe'] = (origin == 2) * 1.0
    dataset['Japan'] = (origin == 3) * 1.0
    print(dataset.tail())
    return dataset


def split_data(dataset, inspect_data=False):
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)
    if inspect_data:
        sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')
        plt.show()
    train_stats = train_dataset.describe()
    train_stats.pop("MPG")
    train_stats = train_stats.transpose()
    if inspect_data:
        print(train_stats)
    return train_dataset, test_dataset, train_stats


def split_features_from_labels(train_dataset, test_dataset):
    train_labels = train_dataset.pop('MPG')
    test_labels = test_dataset.pop('MPG')
    return train_labels, test_labels


def norm(x, train_stats):
    # It is good practice to normalize features with different scales and ranges
    return(x - train_stats['mean']) / train_stats['std']


def build_model(train_dataset):
    # The model will have two densely connected hidden layers and an output layer that returns a single,
    #  continuous value
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model


def try_out_model(model, normed_train_data):
    example_batch = normed_train_data[:10]
    example_result = model.predict(example_batch)
    print(example_result)


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 100 == 0:
            print(' ')
        print('.', end='')


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mae'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'], label='Validation Error')
    plt.ylim([0, 5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mse'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'], label='Validation Error')
    plt.ylim([0, 20])
    plt.legend()

    plt.show()


def plot_predictions(test_labels, test_predictions):
    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True Value [MPG]')
    plt.ylabel('Predictions [MPG]')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0, plt.xlim()[1]])
    plt.ylim([0, plt.ylim()[1]])
    _ = plt.plot([-100, 100], [-100, 100])
    plt.show()


def main():
    print('Getting dataset and importing it to Pandas')
    dataset = get_dataset()

    print('Cleaning dataset by removing unknown values and one-hotting categorical fields')
    dataset = clean_data(dataset)

    print('Splitting data into training and testing and plotting a kernel density estimate')
    train_dataset, test_dataset, train_stats = split_data(dataset, False)

    print('Separating the target (expected) value from the features')
    train_labels, test_labels = split_features_from_labels(train_dataset, test_dataset)

    print('Normalizing the data')
    normed_train_data = norm(train_dataset, train_stats)
    normed_test_data = norm(test_dataset, train_stats)

    print('Building the model')
    model = build_model(train_dataset)
    print(model.summary())

    print('Using the EarlyStopping callback to stop training when the validation score doesn\'t improve')
    # the patience parameter is the amount of epochs to check for improvement
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)

    print('Trying out the model with 10 examples')
    try_out_model(model, normed_train_data)
    print('Training the model')
    history = model.fit(normed_train_data,
                        train_labels,
                        epochs=EPOCHS,
                        validation_split=0.2,
                        verbose=0,
                        callbacks=[early_stop, PrintDot()])
    print('Plotting history')
    plot_history(history)

    print('Evaluating the model on the test data')
    loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)
    print('Testing set Mean Abs Error: {:5.2f} MPG'.format(mae))

    print('Making predictions')
    test_predictions = model.predict(normed_test_data).flatten()
    plot_predictions(test_labels, test_predictions)

    print('Plotting error distribution')
    error = test_predictions - test_labels
    plt.hist(error, bins=25)
    plt.xlabel("Prediction Error [MPG]")
    _ = plt.ylabel('Count')
    plt.show()


main()
