import json

from LatexProducer import get_data
from operator import itemgetter
from itertools import groupby
import matplotlib.pyplot as plt

from data_util import plot_colors
from data_util.data_saver import check_directory


def get_function_json():
    with open('./settings/function_definitions.json', 'r') as f:
        return json.load(f)


def append_dict_to_results_list(dict_in, results_in):
    results_names = [x[0] for x in results_in]

    for idx, name in enumerate(results_names):
        target_dict = [list(d.values())[1:-1] for d in dict_in if d['name'] == name][0]
        results_in[idx].extend(target_dict)

    return results_in


def get_grouper(group_type):
    if group_type == 'dim':
        return group_results_by_dimensionality
    if group_type == 'sep':
        return group_results_by_nonseparable
    if group_type == 'mod':
        return group_results_by_unimodal
    if group_type == 'con':
        return group_results_by_convex
    if group_type == 'dif':
        return group_results_by_differentiable
    if group_type == 'shape':
        return group_results_by_shape
    return False


def group_results_by_nth_col(data, n, key):
    group1 = [d for d in data if d[n] == key]
    group2 = [d for d in data if d[n] != key]

    group1.extend(group2)

    return group1


def group_results_by_shape(data):
    plate = [d for d in data if d[-1] == 'Plate-shaped']    # 5 functions
    bowl = [d for d in data if d[-1] == 'Bowl-shaped']    # 9 functions
    valley = [d for d in data if d[-1] == 'Valley-shaped']    # 3 functions
    many_local_minima = [d for d in data if d[-1] == 'Many Local Minima']    # 19 functions
    steep_drop = [d for d in data if d[-1] == 'Steep Drop']    # 4 functions
    asymmetric = [d for d in data if d[-1] == '']    # 17 functions

    plate.extend(bowl)
    plate.extend(valley)
    plate.extend(many_local_minima)
    plate.extend(steep_drop)
    plate.extend(asymmetric)

    return plate


def group_results_by_dimensionality(data):
    return group_results_by_nth_col(data, 6, "N")


def group_results_by_nonseparable(data):
    return group_results_by_nth_col(data, 7, "True")


def group_results_by_unimodal(data):
    return group_results_by_nth_col(data, 8, "True")


def group_results_by_convex(data):
    return group_results_by_nth_col(data, 9, "True")


def group_results_by_differentiable(data):
    return group_results_by_nth_col(data, 10, "True")


def sort_results_by_nth_col(data, n):
    return sorted(data, key=itemgetter(n))


def sort_results_by_mse_index(data):
    return sort_results_by_nth_col(data, 5)


def sort_results_by_nn_mse(data):
    return sort_results_by_nth_col(data, 2)


def sort_results_by_lr_mse(data):
    return sort_results_by_nth_col(data, 4)


def extract_fcn_and_nn_col(data, n):
    functions = []
    mse_index = []
    for x in data:
        functions.append(x[0])
        mse_index.append(x[n])

    return functions, mse_index


def extract_fcn_and_mse_index(data):
    return extract_fcn_and_nn_col(data, 5)


def extract_fcn_and_nn_mse(data):
    return extract_fcn_and_nn_col(data, 2)


def extract_fcn_and_lr_mse(data):
    return extract_fcn_and_nn_col(data, 4)


def plot_fcns(fcn, mse_index, dest, xlabel='MSE Index'):
    plt.figure(figsize=(12, 10))
    plt.barh(fcn, mse_index, height=0.75, color=plot_colors.BLUE, linewidth=1.1)
    plt.xlabel(xlabel)
    plt.ylabel('Function')
    plt.title('{} per Function'.format(xlabel))
    plt.grid()
    plt.savefig(dest, format='png')
    plt.show()


if __name__ == '__main__':
    function_dict = get_function_json()
    results = get_data()
    results = append_dict_to_results_list(function_dict, results)

    plot_target = './results/mse_index_plots/'
    check_directory(plot_target)

    prop = ''

    grouper = get_grouper(prop)

    results_by_msei = sort_results_by_mse_index(results)
    if grouper:
        results_by_msei = grouper(results_by_msei)
    fcns, mse_indices = extract_fcn_and_mse_index(results_by_msei)
    plot_fcns(fcns, mse_indices, plot_target + prop + 'mse_sorted.png')

    results_by_nn_mse = sort_results_by_nn_mse(results)
    if grouper:
        results_by_nn_mse = grouper(results_by_nn_mse)
    fcns, nn_mse = extract_fcn_and_nn_mse(results_by_nn_mse)
    plot_fcns(fcns, nn_mse, plot_target + prop + 'nn_mse_sorted.png', 'NN MSE')

    results_by_lr_mse = sort_results_by_lr_mse(results)
    if grouper:
        results_by_lr_mse = grouper(results_by_lr_mse)
    fcns, lr_mse = extract_fcn_and_lr_mse(results_by_lr_mse)
    plot_fcns(fcns, lr_mse, plot_target + prop + 'lr_mse_sorted.png', 'LR MSE')
