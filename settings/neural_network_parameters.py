from settings import function_params, optimizers


class NeuralNetworkParameters:

    def __init__(self, num_epochs, num_samples, function, optimizer, hidden_neurons):
        self.num_epochs = num_epochs
        self.num_samples = num_samples
        self.function_definition, self.x_range, self.y_range, self.function_name = parse_function(function)
        self.optimizer, self.learning_rate, self.optimizer_name = self.parse_optimizer(optimizer)
        self.hidden_neurons = hidden_neurons

    @staticmethod
    def parse_optimizer(optimizer):
        if optimizer == 'SGD':
            return optimizers.sgd_optimizer()
        if optimizer == 'Adam':
            return optimizers.adam_optimizer()


def parse_function(function):
    if function == 'ackley':
        return function_params.ackley()
    if function == 'ackley_n2':
        return function_params.ackley_n2()
    if function == 'ackley_n3':
        return function_params.ackley_n3()
    if function == 'ackley_n4':
        return function_params.ackley_n4()
    if function == 'adjiman':
        return function_params.adjiman()
    if function == 'alpine_n1':
        return function_params.alpine_n1()
    if function == 'alpine_n2':
        return function_params.alpine_n2()
    if function == 'beale':
        return function_params.beale()
    if function == 'bartels_conn':
        return function_params.bartels_conn()
    if function == 'bird':
        return function_params.bird()
    if function == 'bohachevsky_n1':
        return function_params.boha_n1()
    if function == 'bohachevsky_n2':
        return function_params.boha_n2()
    if function == 'booth':
        return function_params.booth()
    if function == 'brent':
        return function_params.brent()
    if function == 'brown':
        return function_params.brown()
    if function == 'bukin_n6':
        return function_params.bukin_n6()
    if function == 'cross_in_tray':
        return function_params.cross_in_tray()
    if function == 'deckkers_aarts':
        return function_params.deckkers_arts()
    if function == 'drop_wave':
        return function_params.drop_wave()
    if function == 'easom':
        return function_params.easom()
    if function == 'egg_crate':
        return function_params.egg_crate()
    if function == 'exponential':
        return function_params.exponential()
    return function_params.ackley_n2()
