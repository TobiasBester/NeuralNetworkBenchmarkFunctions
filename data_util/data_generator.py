import numpy as np
import pandas as pd
from pyDOE2 import lhs
from sklearn.preprocessing import MinMaxScaler


def split_and_scale_dims(data, minmax_range):
    x = data.reshape(-1, 1)
    scaler = MinMaxScaler((minmax_range[0], minmax_range[1]))
    return scaler.fit_transform(x)[:, 0]


def generate_random_dataset(x_range, y_range, num_samples, benchmark_func, seed):
    data = lhs(2, num_samples, random_state=seed)

    x = split_and_scale_dims(data[:, 0], x_range)
    y = split_and_scale_dims(data[:, 1], y_range)

    func = benchmark_func(x, y)

    normed_func = normalize_data(func)

    data = {'x': x, 'y': y, 'func': normed_func}

    return pd.DataFrame().from_dict(data)


def generate_regularly_spaced_dataset(x_range, y_range, num_samples, benchmark_func):
    x = np.linspace(start=x_range[0], stop=x_range[1], num=num_samples)
    y = np.linspace(start=y_range[0], stop=y_range[1], num=num_samples)
    func = benchmark_func(x, y)

    data = {'x': x, 'y': y, 'func': func}

    return pd.DataFrame().from_dict(data)


def normalize_data(data, scale=1):
    f_min = min(data)
    f_max = max(data)
    f_range = f_max - f_min

    return ((data - f_min) / f_range) * scale


def generate_uniform_random_dataset(x_range, y_range, num_samples, benchmark_func):
    x = np.random.uniform(x_range[0], x_range[1], num_samples)
    y = np.random.uniform(y_range[0], y_range[1], num_samples)
    func = benchmark_func(x, y)

    normed_func = normalize_data(func)

    data = {'x': x, 'y': y, 'func': normed_func}

    return pd.DataFrame().from_dict(data)

