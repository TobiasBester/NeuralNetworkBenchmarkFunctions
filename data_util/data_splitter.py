import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.mplot3d import Axes3D

from data_util.dataset_group import DatasetGroup


def split_data(dataset, benchmark_func, inspect_data=False):

    train_dataset = dataset.sample(frac=0.75)
    test_dataset = dataset.drop(train_dataset.index)

    train_stats = train_dataset.describe()

    if inspect_data:
        sns.lmplot(data=train_dataset, x='x', y='func')
        plt.show()

        plot_3d_graph(train_dataset['x'], train_dataset['y'], benchmark_func)

        print('Training dataset statistics')
        print(train_stats)

    train_labels, test_labels = split_features_from_labels(train_dataset, test_dataset)

    return DatasetGroup(
        train_dataset,
        test_dataset,
        train_labels,
        test_labels
    )


def plot_3d_graph(x, y, benchmark_func):
    plt.figure()
    ax = plt.axes(projection='3d')

    x, y = np.meshgrid(x, y)
    func = benchmark_func(x, y)

    ax.plot_surface(x, y, func, linewidth=1)

    plt.show()


def split_features_from_labels(train_dataset, test_dataset):
    train_labels = train_dataset.pop('func')
    test_labels = test_dataset.pop('func')
    return train_labels, test_labels
