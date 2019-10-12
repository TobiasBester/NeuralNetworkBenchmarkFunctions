from tensorflow import keras
from tensorflow.keras import layers


def build_model(train_dataset, optimizer, hidden_neurons, show_summary=False):
    model = keras.Sequential([
        layers.Dense(hidden_neurons, input_shape=[len(train_dataset.keys())]),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    # -0.00731060634566405
    if show_summary:
        print(model.summary())

    keras.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layers=True)

    return model
