from data_util.latex_factory import generate_table


def remove_newline(data):
    for x in data:
        x[-1] = x[-1][:-2]


def ensure_precision(data):
    epochs_precision = 2
    mse_precision = 6
    stdev_precision = 4
    mse_index_precision = 6

    for x in data:
        x[1] = round(float(x[1]), epochs_precision)
        x[2] = round(float(x[2]), mse_precision)
        x[3] = round(float(x[3]), stdev_precision)
        x[4] = round(float(x[4]), mse_precision)
        x[5] = round(float(x[5]), mse_index_precision)


def data_to_dict(data):
    return {x[0]: x[1:] for x in data}


def sort_dict_keys(data):
    return {key: data[key] for key in sorted(data)}


def dict_to_list(data):
    [data[x].insert(0, x) for x in data]
    return [data[key] for key in data]


def latex():
    file_name = './results/comparer_results/results.txt'

    f = open(file_name, 'r')

    data = [x.split('|') for x in f.readlines()]

    remove_newline(data)
    ensure_precision(data)
    data = data_to_dict(data)
    data = sort_dict_keys(data)
    data = dict_to_list(data)

    generate_table(data)


latex()
