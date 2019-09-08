from data_util.latex_factory import generate_table


def latex():
    file_name = './results/comparer_results/results.txt'

    f = open(file_name, 'r')

    data = [x.split('|') for x in f.readlines()]

    generate_table(data)


latex()
