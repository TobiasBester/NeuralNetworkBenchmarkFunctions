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

# TODO: a) Increase number of hidden neurons
# TODO: a) Change Compare runner to use updated Runners
# TODO: a) Normalize f(x, y) range to [0, 1] or [-10, 10] or whatever
# TODO: b) Investigate Linear Regression from scikit-learn
# TODO: b) Set different seeds for neural network runs (up to 10 and then use the Mean MSE)
# TODO: c) Look for more functions
# TODO: d) Set NNMSE/LRMSE to something that falls at 0 if equal (in Excel)
# TODO: d) Save output to Latex table
# TODO: e) Plot true surface against NN predicted surface
