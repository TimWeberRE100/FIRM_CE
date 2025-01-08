from typing import Dict, List
import numpy as np
from numba import njit, prange

from firm_ce import ModelConfig, Scenario 

class Solution:
    def __init__(self, x) -> None:
        self.x = x            

class Solver:
    def __init__(self, config: ModelConfig, scenario: Scenario) -> None:
        self.config = config
        self.decision_x = None
        self.upper_bounds = None
        self.lower_bounds = None
        self.solution = Solution(self.decision_x)
        self.evaluated=False      

        self.demand = np.array([scenario.nodes[idx].demand_data for idx in range(0,max(scenario.nodes))], dtype=np.float64)
        self.intervals, self.nodes = self.demand.shape
        self.years = scenario.final_year - scenario.first_year      

    def _reliability(self):
        pass

    def _single_time_objective(self):
        pass

    def evaluate(self):
        self.lcoe, self.penalties = self._single_time_objective()
        self.evaluated=True 

    def evaluate(self) -> None:
        if self.config.type not in ['single_time','capacity_expansion']:
            raise Exception("Model type in config must be 'single_time' or 'capacity_expansion")

        if self.config.type == 'single_time':
            self.solution = _single_time()
        elif self.config.type == 'capacity_expansion':
            self.solution = self._capacity_expansion()

@njit(parallel=True)
def parallel_objective_wrapper(xs):
    result = np.empty(xs.shape[1], dtype=np.float64)
    for i in prange(xs.shape[1]):
        result[i] = objective(xs[:,i])
    return result

def objective(solver: Solver):
    solver.evaluate()