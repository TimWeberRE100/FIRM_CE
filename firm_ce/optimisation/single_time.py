import numpy as np
import time

#from firm_ce.system.topology import get_transmission_flows_t
from firm_ce.common.constants import JIT_ENABLED, NUM_THREADS, PENALTY_MULTIPLIER
from firm_ce.system.costs import calculate_costs
from firm_ce.system.components import Fleet
from firm_ce.system.topology import Network
from firm_ce.system.energybalance import ScenarioParameters, EnergyBalance, FleetCapacities, IntervalMemory
import firm_ce.common.helpers as helpers

from firm_ce.optimisation.balancing import balance_for_period

if JIT_ENABLED:
    from numba import float64, boolean, njit, prange, set_num_threads
    from numba.experimental import jitclass

    set_num_threads(int(NUM_THREADS))

    solution_spec = [
        ('x', float64[:]),
        ('evaluated', boolean),
        ('lcoe', float64),
        ('penalties', float64),

        # Static jitclass instances
        ('static', ScenarioParameters.class_type.instance_type),

        # Dynamic jitclass instances
        ('fleet', Fleet.class_type.instance_type),
        ('network', Network.class_type.instance_type),

        
        #('energy_balance',EnergyBalance.class_type.instance_type),
        #('fleet_capacities',FleetCapacities.class_type.instance_type),
        #('interval_memory',IntervalMemory.class_type.instance_type),
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
                #energy_balance
                ) -> None:
        self.x = x  
        self.evaluated=False   
        self.lcoe = 0.0
        self.penalties = 0.0

        # These are static jitclass instances. It is unsafe to modify these
        # within a worker process of the optimiser
        self.static = static 
        """ self.fleet_static = fleet
        self.network_static = network """

        # These are dynamic jitclass instances. They are safe to modify
        # within a worker process of the optimiser
        self.network = network.create_dynamic_copy() # Includes static reference to data
        self.fleet = fleet.create_dynamic_copy(self.network.nodes, self.network.minor_lines) # Includes static reference to data
        #self.energy_balance = energy_balance.create_dynamic_copy() # Includes dynamic copy of residual_load
        
        self.fleet.build_capacities(x)

        self.fleet.allocate_memory(self.static.intervals_count)
        self.network.allocate_memory()

        """ self.energy_balance.allocate_memory(
            self.static.node_count, 
            self.static.intervals_count,
            len(self.fleet.storages),
            self.fleet.get_generator_unit_type_count("flexible"),
        ) """
        
        # Trying to replace the below with a dynamic fleet
        """ self.fleet_capacities = FleetCapacities(False)
        self.interval_memory = IntervalMemory(
            False, 
            self.static.node_count,
            len(self.fleet.storages),
            self.fleet.get_generator_unit_type_count("flexible"),
        )

        # Initialise the dynamic jitclass instances
        self.fleet_capacities.load_data(self.fleet, self.static.node_count) # ADD SORTED ORDER LOADING TO METHOD?
        self.fleet_capacities.build_capacities(self.fleet, x) """

    def balance_residual_load(self) -> bool: 
        # Can I get away with avoiding the 2d arrays during the DE?
        # Just generate them when calculating statistics?
        # Update the relevate trace arrays at the end of each time interval
        # if the 'statistics_flag' Solution argument is True
        #self.energy_balance.initialise_stored_energy(self.fleet.storages)
        self.fleet.initialise_stored_energies()

        for year in range(self.static.year_count):
            first_t, last_t = self.static.get_year_t_boundaries(year)
            self.fleet.initialise_flexible_annual_limits(year, first_t)

            """ balance_for_period(
                first_t,
                last_t,
                True,
                self
            ) 

            annual_unserved_energy = self.network.calculate_unserved_energy(first_t, last_t) * self.static.resolution
            
            if not self.static.check_reliability_constraint(year, annual_unserved_energy):
                self.penalties += PENALTY_MULTIPLIER
                return False """
        return True

    def objective(self):
        self.network.update_residual_loads(
            self.fleet.generators,
            self.static.intervals_count,
        )        

        if not self.balance_residual_load(): 
            pass ##### DEBUG    
            #return self.lcoe, self.penalties # End early if reliability constraint breached
        
        # self.apportion_nodal_storage() # Add traces2d to fleet_capacities?
        # self.calculate_annual_generation()
        # cost, _, _, _ = calculate_costs()

        # line_losses = self.network.transmission_flows_abs.sum(axis=0) 
        # for order, line in self.network.lines.items():
        #   line_losses[order] *= line.loss_factor
        # total_losses = loss.sum() * self.resolution / self.years

        # lcoe = cost / np.abs(self.static.energy - total_losses) / 1000 # $/MWh

        lcoe = sum(self.x) #### DEBUG
        pen_deficit = lcoe*10000 - 400 #### DEBUG
        
        return lcoe, pen_deficit 

    def evaluate(self):
        self.lcoe, self.penalties = self.objective()
        self.evaluated=True 
        return self

@njit(parallel=True) # ADD FASTMATH FLAG AND TEST?
def parallel_wrapper(xs, 
                    static,
                    fleet,
                    network,
                    #energy_balance
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
                       #energy_balance
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
                           #energy_balance
                           ):
    start_time = time.time()
    result = parallel_wrapper(xs,
                             static,
                             fleet,
                             network,
                             #energy_balance
                             )  
    end_time = time.time()  
    print(f"Objective time: {(end_time-start_time)/xs.shape[1]:.4f} seconds")
    print(f"Iteration time: {(end_time-start_time):.4f} seconds")
    return result[0,:]