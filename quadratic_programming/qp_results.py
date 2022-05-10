import numpy as np

class Results:

    def __init__(self) -> None:
        self.iterations = list()
        self.success = False
    
    @property
    def num_iterations(self):
        return len(self.iterations)
    
    @property
    def x_solution(self):
        return self.iterations[self.num_iterations-1].xk

    @property
    def lmda_solution(self):
        return self.iterations[self.num_iterations-1].lmda
    
    def print_iterations(self):
        for it in self.iterations:
            it.pprint()
    
    def print_solution(self):
        print(f"=== Solution ===\nx:\n{self.x_solution}\nMultipliers:\n{self.lmda_solution}")

class IterationResult:
    
    def __init__(self, k : int) -> None:
        self.k              = int(k)
        self.xk             = np.matrix([])
        self.lmda           = np.matrix([])
        self.pk             = np.matrix([])
        self.working_set    = set()
        self.idx_to_work    = None
        self.idx_from_work  = None
        self.alphak         = None
    
    def pprint(self):
        print(f"============ Iteration {self.k} ============")
        for key, value in vars(self).items():
            if key == "k":
                continue
            print(f"{key}:\n{value}")
