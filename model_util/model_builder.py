from tensorflow import keras
from tensorflow.keras import layers


def build_model(train_dataset, optimizer, hidden_neurons, show_summary=False):
    model = keras.Sequential([
        layers.Dense(2, activation='sigmoid', input_shape=[len(train_dataset.keys())]),
        layers.Dense(hidden_neurons, activation='sigmoid'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])

    if show_summary:
        print(model.summary())

    return model
