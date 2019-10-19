from pylatex import Document, LongTabu
from pylatex.utils import bold

from data_util.data_saver import check_directory


def generate_table(data):
    geometry_options = {
        'landscape': True,
        'margin': '1.5in',
        'headheight': '20pt',
        'headsep': '10pt',
        'includeheadfoot': True
    }

    doc = Document(page_numbers=True, geometry_options=geometry_options)

    fmt = 'X[r] X[r] X[r] X[r] X[r] X[r]'
    with doc.create(LongTabu(fmt, spread='0pt')) as data_table:
        header_row1 = ['Function Name', 'NN Mean Num Epochs Run', 'NN MSE', 'NN MSE Stdev', 'LR MSE', 'MSE Index']
        data_table.add_row(header_row1, mapper=[bold])
        data_table.add_hline()
        data_table.add_empty_row()
        data_table.end_table_header()

        for line in data:
            data_table.add_row(line)

    output_path = './results/latex/'
    check_directory(output_path)
    doc.generate_tex(output_path + 'table')
