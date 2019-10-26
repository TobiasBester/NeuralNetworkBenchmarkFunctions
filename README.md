# Neural Network Regression Benchmark Functions

## Prerequisites

- Python 3.6 (not 3.7, otherwise TensorFlow 2.0 won't work)
- pip

## Installation

- Run the following command in the root directory of the project (not tested)

    - `python setup.py`
    
## Usage

The config files `lr_config.ini` and `nn_config.ini` are used to specify the parameters for the LinearRegression and
NeuralNetwork programs respectively. Everything can be adjusted, although changing certain parameters may lead to
suboptimal performance.

The optimizer and function parameter options are all provided and commented out. To change these parameters, uncomment
the option of your choice and ensure that only one of each parameter is uncommented.

Running `python LRRunner.py` runs the Linear Regression program for the specified problem function.

Running `python NNRunner.py` runs the Neural Network program for the specified problem function.

Running `python Comparer.py` runs the NeuralNetwork program and then the LinearRegression program.
