from data_util.dataset_group import DatasetGroup
from data_util.data_plotter import plot_3d_graph


def split_data_for_nn(dataset, benchmark_func, show_generated_data=True):
    train_dataset, test_dataset = split_data(dataset, benchmark_func, show_generated_data)

    train_labels = train_dataset.pop('func')
    test_labels = test_dataset.pop('func')

    return DatasetGroup(
        train_dataset,
        test_dataset,
        train_labels,
        test_labels
    )


def split_data_for_lr(dataset, benchmark_func, show_generated_data=True):
    train_dataset, test_dataset = split_data(dataset, benchmark_func, show_generated_data)

    train_input = list(zip(train_dataset['x'], train_dataset['y']))
    train_labels = list(train_dataset['func'])

    test_input = list(zip(test_dataset['x'], test_dataset['y']))
    test_labels = list(test_dataset['func'])

    return DatasetGroup(
        train_input,
        test_input,
        train_labels,
        test_labels
    )


def split_data(dataset, benchmark_func, show_generated_data=True):
    train_dataset = dataset.sample(frac=0.75, random_state=123)
    test_dataset = dataset.drop(train_dataset.index)

    train_stats = train_dataset.describe()

    if show_generated_data:
        plot_3d_graph(train_dataset['x'], train_dataset['y'], benchmark_func, 'Generated data')

        print('Training dataset statistics')
        print(train_stats)

    return train_dataset, test_dataset
