from __future__ import absolute_import, division, print_function

import pandas as pd
from tensorflow import keras


def get_dataset():
    # Get autoMPG dataset
    dataset_path = keras.utils.get_file("auto-mpg.data",
                                        "https://archive.ics.uci.edu/ml/machine-learning-databases/auto"
                                        "-mpg/auto-mpg.data")

    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                    'Acceleration', 'Model Year', 'Origin']
    # import the dataset to pandas
    raw_dataset = pd.read_csv(dataset_path,
                              names=column_names,
                              na_values='?',
                              comment='\t',
                              sep=' ',
                              skipinitialspace=True)
    dataset = raw_dataset.copy()
    print(dataset.tail())

    return dataset


def clean_data(dataset):
    # drop the rows with unknown values
    dataset = dataset.dropna()

    # convert the "Origin" column from categorical to numerical via one-hotting
    #   one-hotting is representing categorical variables as binary vectors
    #       this first requires that the categorical values be mapped to integer values
    #       then each integer value is represented as a binary vector that is all zeros except the index of the integer,
    #        which is a 1
    origin = dataset.pop('Origin')
    dataset['USA'] = (origin == 1) * 1.0
    dataset['Europe'] = (origin == 2) * 1.0
    dataset['Japan'] = (origin == 3) * 1.0
    print(dataset.tail())


def main():
    dataset = get_dataset()
    clean_data(dataset)


main()
