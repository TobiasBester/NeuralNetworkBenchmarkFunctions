from data_util.latex_factory import generate_table


def remove_newline(data):
    for x in data:
        x[-1] = x[-1][:-2]


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
    data = data_to_dict(data)
    data = sort_dict_keys(data)
    data = dict_to_list(data)

    generate_table(data)


latex()
