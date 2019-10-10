from os import path, makedirs


def save_generated_nn_data_to_file(function_name, dataset_group):
    data_path = './generated_data'
    check_directory(data_path)

    file_name = "%s/%s__data.txt" % (data_path, function_name)

    f = open(file_name, "w+")
    f.write("Training Set\n")

    save_generated_data_to_file(
        list(dataset_group.train_dataset['x']),
        list(dataset_group.train_dataset['y']),
        list(dataset_group.train_labels),
        f)

    f.write("Test Set\n")
    save_generated_data_to_file(
        list(dataset_group.test_dataset['x']),
        list(dataset_group.test_dataset['y']),
        list(dataset_group.test_labels),
        f)

    print("Saved generated data to", file_name)

    f.close()


def save_generated_lr_data_to_file(function_name, dataset_group):
    data_path = './generated_data'
    check_directory(data_path)

    file_name = "%s/%s__data.txt" % (data_path, function_name)

    f = open(file_name, "w+")
    f.write("Training Set\n")

    save_generated_data_to_file(
        [x for (x, y) in dataset_group.train_dataset],
        [y for (x, y) in dataset_group.train_dataset],
        dataset_group.train_labels,
        f)

    f.write("Test Set\n")
    save_generated_data_to_file(
        [x for (x, y) in dataset_group.test_dataset],
        [y for (x, y) in dataset_group.test_dataset],
        list(dataset_group.test_labels),
        f)

    print("Saved generated data to", file_name)

    f.close()


def save_generated_data_to_file(x, y, func, f):
    f.write("x    y    f(x, y)\n")

    for i in range(0, len(x)):
        f.write("%s %s %s\n" % (x[i], y[i], func[i]))


def save_combined_results_to_file(func_name, nn_mse, nn_mse_stdev, lr_mse, mse_index):
    file_path = './results/comparer_results'
    check_directory(file_path)

    file_name = file_path + '/results.txt'

    f = open(file_name, 'a+')

    line = "{}|{}|{}|{}|{}\n".format(func_name, nn_mse, nn_mse_stdev, lr_mse, mse_index)

    f.write(line)
    f.close()


def check_directory(results_path):
    if not path.exists(results_path):
        makedirs(results_path)
