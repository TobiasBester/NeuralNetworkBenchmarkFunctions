from settings.dataset_parameters import parse_function


class LinearRegressionParameters:

    def __init__(self, num_samples, function):
        self.num_samples = num_samples
        self.function_definition, self.x_range, self.y_range, self.function_name = parse_function(function)
