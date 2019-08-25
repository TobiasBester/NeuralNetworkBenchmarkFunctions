from settings import function_params, optimizers


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
        self.optimizer, self.learning_rate, self.optimizer_name = optimizer
        self.show_model_details = show_model_details
        self.show_training = show_training

    @staticmethod
    def parse_optimizer(optimizer):
        if optimizer == 'SGD':
            return optimizers.sgd_optimizer()
        if optimizer == 'Adam':
            return optimizers.adam_optimizer()

# TODO: 0) Change main programs for NN and LR to a callable function, so that random seeds can be set
# a) Increase number of Num Samples
# TODO: a) Normalize f(x, y) range to [0, 1] or [-10, 10] or whatever
# TODO: b) Investigate Linear Regression from scikit-learn
# TODO: b) Set different seeds for neural network runs (up to 10 and then use the Mean MSE)
# TODO: c) Experiment with Sigmoid in place of ReLU
# TODO: c) Save sample points for each function somewhere 1/2
# TODO: c) Look for more functions
# TODO: d) Set NNMSE/LRMSE to something that falls at 0 if equal
# TODO: d) Save output to Latex table
# TODO: e) Plot true surface in 3D
# TODO: e) Plot true surface against NN predicted surface
