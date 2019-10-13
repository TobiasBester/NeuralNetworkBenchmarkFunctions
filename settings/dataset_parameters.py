from settings import function_params


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
    if function == 'egg_holder':
        return function_params.egg_holder()
    if function == 'exponential':
        return function_params.exponential()
    if function == 'goldstein_price':
        return function_params.goldstein_price()
    if function == 'griewank':
        return function_params.griewank()
    if function == 'himmelblau':
        return function_params.himmelblau()
    if function == 'holder_table':
        return function_params.holder_table()
    if function == 'keane':
        return function_params.keane()
    if function == 'leon':
        return function_params.leon()
    if function == 'levi_n13':
        return function_params.levi_n13()
    if function == 'matyas':
        return function_params.matyas()
    if function == 'mccormick':
        return function_params.mccormick()
    if function == 'michalewicz':
        return function_params.michalewicz()
    if function == 'periodic':
        return function_params.periodic()
    if function == 'qing':
        return function_params.qing()
    if function == 'rastrigin':
        return function_params.rastrigin()
    if function == 'ridge':
        return function_params.ridge()
    if function == 'rosenbrock':
        return function_params.rosenbrock()
    if function == 'salomon':
        return function_params.salomon()
    if function == 'schaffer_n2':
        return function_params.schaffer_n2()
    if function == 'schaffer_n3':
        return function_params.schaffer_n3()
    if function == 'schwefel_220':
        return function_params.schwefel_220()
    if function == 'schwefel_222':
        return function_params.schwefel_222()
    if function == 'schwefel_223':
        return function_params.schwefel_223()
    if function == 'shubert_3':
        return function_params.shubert_3()
    if function == 'shubert':
        return function_params.shubert()
    if function == 'sphere':
        return function_params.sphere()
    if function == 'styblinski_tang':
        return function_params.styblinski_tang()
    if function == 'sum_squares':
        return function_params.sum_squares()
    if function == 'three_hump_camel':
        return function_params.three_hump_camel()
    if function == 'xin_she_yang_n2':
        return function_params.xin_she_yang_n2()
    if function == 'xin_she_yang_n3':
        return function_params.xin_she_yang_n3()
    if function == 'xin_she_yang_n4':
        return function_params.xin_she_yang_n4()
    if function == 'zakharov':
        return function_params.zakharov()
    raise Exception('Function could not be found')


class DatasetParameters:

    def __init__(self,
                 seed,
                 num_samples,
                 show_true_function,
                 show_generated_data,
                 save_generated_data,
                 show_predicted_vs_true,
                 function):
        self.seed = seed
        self.num_samples = num_samples
        self.show_true_function = show_true_function
        self.show_generated_data = show_generated_data
        self.save_generated_data = save_generated_data
        self.show_predicted_vs_true = show_predicted_vs_true
        self.function_definition, self.x_range, self.y_range, self.function_name = parse_function(function)
