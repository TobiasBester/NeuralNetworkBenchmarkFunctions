from settings import optimizers


class NeuralNetworkParameters:

    def __init__(self,
                 num_seeds,
                 starting_seed,
                 num_epochs,
                 hidden_neurons,
                 optimizer,
                 show_model_details,
                 show_training):

        self.num_seeds = num_seeds
        self.starting_seed = starting_seed
        self.num_epochs = num_epochs
        self.hidden_neurons = hidden_neurons
        self.optimizer, self.learning_rate, self.optimizer_name = self.parse_optimizer(optimizer)
        self.show_model_details = show_model_details
        self.show_training = show_training

    @staticmethod
    def parse_optimizer(optimizer):
        if optimizer == 'SGD':
            return optimizers.sgd_optimizer()
        if optimizer == 'Adam':
            return optimizers.adam_optimizer()
