
class Options:

    def __init__(self, verbosity : int, x_init = None, working_set_init = set(), decimal_precision=5) -> None:
        self.verbosity         = verbosity
        self.x_init            = x_init
        self.working_set_init  = working_set_init
        self.decimal_precision = decimal_precision
