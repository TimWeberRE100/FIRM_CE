import numpy as np
from typing import List
import time

from firm_ce.system.topology import get_transmission_flows_t
from firm_ce.common.constants import JIT_ENABLED, EPSILON_FLOAT64, NUM_THREADS
from firm_ce.system.costs import calculate_costs
from firm_ce.optimisation.numba_classes import *
import firm_ce.common.helpers as helpers

if JIT_ENABLED:
    from numba import float64, int64, boolean, njit, prange, set_num_threads
    from numba.experimental import jitclass

    set_num_threads(int(NUM_THREADS))

    solution_spec = [
        ('x', float64[:]),
        ('evaluated', boolean),
        ('lcoe', float64),
        ('penalties', float64),

        ('scenario', Scenario_JIT.class_type.instance_type),
        ('fleet', Fleet_JIT.class_type.instance_type),
        ('buses', Buses_JIT.class_type.instance_type),
        ('network', Network_JIT.class_type.instance_type),
    ]
else:
    def jitclass(spec):
        def decorator(cls):
            return cls
        return decorator
    
    def njit(func=None, **kwargs):
        if func is not None:
            return func
        def wrapper(f):
            return f
        return wrapper
    
    def prange(x):
        return range(x)
    
    solution_spec = []

@jitclass(solution_spec)
class Solution_SingleTime:
    def __init__(self, 
                x, 
                scenario,
                fleet,
                buses,
                network) -> None:
        
        self.x = x  
        self.evaluated=False   
        self.lcoe = 0.0
        self.penalties = 0.0

        self.scenario = scenario
        self.fleet = fleet
        self.buses = buses
        self.network = network
        
        self._initialise_capacities(x)
        self._update_traces() # Multiply availability by capacity

    def _initialise_capacities(self, x):
        # Iterate through fleet capacities
        # Iterate through line capacities
        return None

    def _objective(self):
        start_time = time.time()

        deficit, TFlowsAbs = self._transmission_balancing()
        pen_deficit = np.maximum(0., deficit.sum() * self.resolution / self.years - self.allowance) * 1000000

        end_time = time.time()
        print(f"Transmission time: {end_time-start_time:.4f} seconds")

        self._apportion_nodal_storage()
        end_time2 = time.time()
        print(f"Storage apportion time: {end_time2-end_time:.4f} seconds")

        self._calculate_annual_generation()
        cost, _, _, _ = calculate_costs(self)

        loss = TFlowsAbs.sum(axis=0) * self.TLoss
        self.loss = loss.sum() * self.resolution / self.years

        lcoe = cost / np.abs(self.energy - self.loss) / 1000 # $/MWh
        
        print("LCOE: ", lcoe, pen_deficit, deficit.sum() / self.MLoad.sum(), self.GFlexible_annual)
        exit()
        return lcoe, pen_deficit

    def evaluate(self):
        self.lcoe, self.penalties = self._objective()
        self.evaluated=True 
        return self

@njit(parallel=True)
def parallel_wrapper(xs,
                     scenario,
                     fleet,
                     buses,
                     network):
    """
    parallel_wrapper, but also returns LCOE and penalty seperately
    """
    n_points = xs.shape[1]
    result = np.zeros((3, n_points), dtype=np.float64)
    for j in prange(n_points):
        xj = xs[:, j]
        sol = Solution_SingleTime(xj,
                                  scenario,
                                  fleet,
                                  buses,
                                  network)
        sol.evaluate()
        result[0, j] = sol.lcoe + sol.penalties
        result[1, j] = sol.lcoe
        result[2, j] = sol.penalties
    return result

@njit
def initialise_single_time(xs,
                            scenario,
                            fleet,
                            buses,
                            network):
    

    result = parallel_wrapper(xs,
                            scenario,
                            fleet,
                            buses,
                            network)
    
    return result[0,:]