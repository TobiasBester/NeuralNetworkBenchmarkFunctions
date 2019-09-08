from pylatex import Document, LongTabu
from pylatex.utils import bold


def generate_table(data):
    geometry_options = {
        "landscape": True,
        "margin": "1.5in",
        "headheight": "20pt",
        "headsep": "10pt",
        "includeheadfoot": True
    }

    doc = Document(page_numbers=True, geometry_options=geometry_options)

    fmt = "X[r] X[r] X[r] X[r]"
    with doc.create(LongTabu(fmt, spread="0pt")) as data_table:
        header_row1 = ["Function Name", "NN MSE", "LR MSE", "MSE Index"]
        data_table.add_row(header_row1, mapper=[bold])
        data_table.add_hline()
        data_table.add_empty_row()
        data_table.end_table_header()

        for line in data:
            data_table.add_row(line)

    doc.generate_tex('./results/latex/table')
