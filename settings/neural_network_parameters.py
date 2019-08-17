from settings import function_params, optimizers


class NeuralNetworkParameters:

    def __init__(self, num_epochs, num_samples, function, optimizer, hidden_neurons):
        self.num_epochs = num_epochs
        self.num_samples = num_samples
        self.function_definition, self.x_range, self.y_range, self.function_name = self.parse_function(function)
        self.optimizer, self.learning_rate, self.optimizer_name = self.parse_optimizer(optimizer)
        self.hidden_neurons = hidden_neurons

    @staticmethod
    def parse_function(function):
        if function == 'ackley_n2':
            return function_params.ackley_n2()
        if function == 'adjiman':
            return function_params.adjiman()
        if function == 'beale':
            return function_params.beale()
        return function_params.ackley_n2()

    @staticmethod
    def parse_optimizer(optimizer):
        if optimizer == 'SGD':
            return optimizers.sgd_optimizer()
        if optimizer == 'Adam':
            return optimizers.adam_optimizer()
