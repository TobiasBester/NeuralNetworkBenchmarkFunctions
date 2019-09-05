import numpy as np
import pandas as pd


def generate_random_dataset(x_range, y_range, num_samples, benchmark_func):
    x = np.random.uniform(x_range[0], x_range[1], num_samples)
    y = np.random.uniform(y_range[0], y_range[1], num_samples)
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


def normalize_data(data, multiplier=100):
    f_min = min(data)
    f_max = max(data)
    f_range = f_max - f_min

    return ((data - f_min) / f_range) * multiplier
