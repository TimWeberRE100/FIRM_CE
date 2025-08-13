import numpy as np
import time

from firm_ce.common.constants import JIT_ENABLED, NUM_THREADS, PENALTY_MULTIPLIER, FASTMATH
from firm_ce.system.components import Fleet
from firm_ce.system.topology import Network
from firm_ce.system.parameters import ScenarioParameters

from firm_ce.optimisation.balancing import balance_for_period

if JIT_ENABLED:
    from numba import njit, prange, set_num_threads
    from numba.core.types import float64, boolean, string
    from numba.experimental import jitclass

    set_num_threads(int(NUM_THREADS))

    solution_spec = [
        ('x', float64[:]),
        ('evaluated', boolean),
        ('lcoe', float64),
        ('penalties', float64),

        # Static jitclass instances
        ('static', ScenarioParameters.class_type.instance_type),
        ('balancing_type', string),

        # Dynamic jitclass instances
        ('fleet', Fleet.class_type.instance_type),
        ('network', Network.class_type.instance_type),
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
class Solution:
    def __init__(self, 
                x, 
                static,
                fleet,
                network,
                balancing_type,
                ) -> None:
        self.x = x  
        self.evaluated=False   
        self.lcoe = 0.0
        self.penalties = 0.0

        # These are static jitclass instances. It is unsafe to modify these
        # within a worker process of the optimiser
        self.static = static 
        self.balancing_type = balancing_type

        # These are dynamic jitclass instances. They are safe to modify
        # some attributes within a worker process of the optimiser
        self.network = network.create_dynamic_copy() # Includes static reference to data
        self.fleet = fleet.create_dynamic_copy(self.network.nodes, self.network.minor_lines) # Includes static reference to data
        self.fleet.build_capacities(x, self.static.resolution)

        self.fleet.allocate_memory(self.static.intervals_count)
        self.network.allocate_memory(self.static.intervals_count)

        self.network.assign_storage_merit_orders(self.fleet.storages)
        self.network.assign_flexible_merit_orders(self.fleet.generators)

    def balance_residual_load(self) -> bool: 
        self.fleet.initialise_stored_energies()        

        for year in range(self.static.year_count):
            first_t, last_t = self.static.get_year_t_boundaries(year)

            self.fleet.initialise_annual_limits(year, first_t)
            
            balance_for_period(
                first_t,
                last_t,
                True,
                self
            ) 

            annual_unserved_energy = self.network.calculate_period_unserved_power(first_t, last_t) * self.static.resolution
            
            # End early if reliability constraint breached for any year
            if not self.static.check_reliability_constraint(year, annual_unserved_energy): 
                self.penalties += (self.static.year_count - year) * annual_unserved_energy * PENALTY_MULTIPLIER
                return False
        return True
    
    def calculate_costs(self) -> float:
        total_costs = 0.0
        years_float = self.static.year_count * self.static.fom_scalar

        self.fleet.calculate_lt_generations(
            self.static.resolution,
        )
        self.network.calculate_lt_flows(
            self.static.resolution,
        )

        for generator in self.fleet.generators.values():
            total_costs += generator.calculate_lt_costs(years_float)

        for storage in self.fleet.storages.values():
            total_costs += storage.calculate_lt_costs(years_float)

        for line in self.network.major_lines.values():
            total_costs += line.calculate_lt_costs(years_float)

        for line in self.network.minor_lines.values():
            total_costs += line.calculate_lt_costs(years_float)

        return total_costs

    def objective(self):        
        reliability_check = self.balance_residual_load()

        if not reliability_check: 
            return self.lcoe, self.penalties # End early if reliability constraint breached
        
        total_costs = self.calculate_costs()

        total_line_losses = self.network.calculate_lt_line_losses()

        lcoe = total_costs / np.abs(sum(self.static.year_energy_demand) - total_line_losses) / 1000 # $/MWh

        return lcoe, self.penalties 

    def evaluate(self):
        self.lcoe, self.penalties = self.objective()
        self.evaluated=True 
        return self

@njit(parallel=True) 
def parallel_wrapper(xs, 
                    static,
                    fleet,
                    network,
                    balancing_type,
                    ):
    """
    parallel_wrapper, but also returns LCOE and penalty seperately
    """
    n_points = xs.shape[1]
    result = np.zeros((3, n_points), dtype=np.float64)
    for j in prange(n_points):
        xj = xs[:, j]
        sol = Solution(xj, 
                       static,
                       fleet,
                       network,
                       balancing_type,
                       )
        sol.evaluate()
        result[0, j] = sol.lcoe + sol.penalties
        result[1, j] = sol.lcoe
        result[2, j] = sol.penalties
    return result

#@njit
def evaluate_vectorised_xs(xs,
                           static,
                           fleet,
                           network,
                           balancing_type,
                           ):
    start_time = time.time()
    result = parallel_wrapper(xs,
                             static,
                             fleet,
                             network,
                             balancing_type,
                             )  
    end_time = time.time()  
    print(f"Objective time: {(end_time-start_time)/xs.shape[1]:.4f} seconds")
    print(f"Iteration time: {(end_time-start_time):.4f} seconds")
    return result[0,:]