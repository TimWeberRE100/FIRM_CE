import numpy as np
import time

from firm_ce.common.constants import JIT_ENABLED, NUM_THREADS, PENALTY_MULTIPLIER, FASTMATH
from firm_ce.system.components import Fleet
from firm_ce.system.topology import Network
from firm_ce.system.parameters import ScenarioParameters
from firm_ce.fast_methods import (
    network_m, line_m,
    generator_m, storage_m, fleet_m,
    static_m
)
from firm_ce.optimisation.balancing import balance_for_period
from firm_ce.common.typing import float64, string, boolean
from firm_ce.common.jit_overload import njit, jitclass, prange

if JIT_ENABLED:
    from numba import set_num_threads
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
        self.network = network_m.create_dynamic_copy(network) # Includes static reference to data
        self.fleet = fleet_m.create_dynamic_copy(fleet, self.network.nodes, self.network.minor_lines) # Includes static reference to data
        fleet_m.build_capacities(self.fleet, x, self.static.resolution)
        network_m.build_capacity(self.network, x)

        fleet_m.allocate_memory(self.fleet, self.static.intervals_count)
        network_m.allocate_memory(self.network, self.static.intervals_count)

        network_m.assign_storage_merit_orders(self.network, self.fleet.storages)
        network_m.assign_flexible_merit_orders(self.network, self.fleet.generators)

    def balance_residual_load(self) -> bool: 
        fleet_m.initialise_stored_energies(self.fleet)        

        for year in range(self.static.year_count):
            first_t, last_t = static_m.get_year_t_boundaries(self.static, year)

            fleet_m.initialise_annual_limits(self.fleet, year, first_t)
            
            balance_for_period(
                first_t,
                last_t,
                True,
                self
            ) 

            annual_unserved_energy = network_m.calculate_period_unserved_power(self.network, first_t, last_t) * self.static.resolution
            
            # End early if reliability constraint breached for any year
            if not static_m.check_reliability_constraint(self.static, year, annual_unserved_energy): 
                self.penalties += (self.static.year_count - year) * annual_unserved_energy * PENALTY_MULTIPLIER
                return False
        return True
    
    def calculate_costs(self) -> float:
        total_costs = 0.0
        years_float = self.static.year_count * self.static.fom_scalar

        fleet_m.calculate_lt_generations(
            self.fleet,
            self.static.resolution,
        )
        network_m.calculate_lt_flows(
            self.network,
            self.static.resolution,
        )

        for generator in self.fleet.generators.values():
            total_costs += generator_m.calculate_lt_costs(generator, years_float, self.static.year_count)
            
        for storage in self.fleet.storages.values():
            total_costs += storage_m.calculate_lt_costs(storage, years_float, self.static.year_count)
            
        for line in self.network.major_lines.values():
            total_costs += line_m.calculate_lt_costs(line, years_float, self.static.year_count)
        
        for line in self.network.minor_lines.values():
            total_costs += line_m.calculate_lt_costs(line,  years_float, self.static.year_count)
        
        return total_costs

    def objective(self):        
        reliability_check = self.balance_residual_load()

        if not reliability_check: 
            return self.lcoe, self.penalties # End early if reliability constraint breached
        
        total_costs = self.calculate_costs()

        total_line_losses = network_m.calculate_lt_line_losses(self.network)

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