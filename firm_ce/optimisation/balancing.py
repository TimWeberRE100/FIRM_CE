import numpy as np
from numpy.typing import NDArray

from firm_ce.common.constants import JIT_ENABLED
from firm_ce.system.topology import Network_InstanceType
from firm_ce.system.components import Fleet_InstanceType
from firm_ce.common.helpers import (
    array_min, 
    array_max_2d_axis1, 
    array_sum_2d_axis0, 
    zero_safe_division
)

if JIT_ENABLED:
    from numba import njit
else:
    def njit(func=None, **kwargs):
        if func is not None:
            return func
        def wrapper(f):
            return f
        return wrapper

@njit 
def balance_with_neighbouring_surplus(interval: int,
                                      network: Network_InstanceType,
                                      ) -> bool:
    network.set_node_fills_and_surpluses('neighbour_surplus')
    network.fill_with_transmitted_surpluses(interval)
    
    if network.check_remaining_netloads():
        return True
    return False

@njit 
def initialise_interval(interval: int,
                        network: Network_InstanceType,
                        fleet: Fleet_InstanceType,
                        resolution: float,) -> None:
    for node in network.nodes.values():
        node.initialise_netload_t(interval)
    
    for generator in fleet.generators.values():
        if generator.check_unit_type('flexible'):
            generator.set_flexible_max_t(interval, resolution)
    
    for storage in fleet.storages.values():
        storage.set_dispatch_max_t(interval, resolution)
    return None

@njit
def balance_for_period(start_t: int, 
                       end_t: int, 
                       precharging_allowed: bool,
                       solution,
                       ) -> None:
    perform_precharge = False
   
    for t in range(start_t, end_t):
        initialise_interval(
            t,
            solution.network,
            solution.fleet,            
            solution.static.resolution
        )

        if not precharging_allowed:
            solution.network.reset_transmission(t)

        if solution.network.check_remaining_netloads():
            continue_balancing_check = balance_with_neighbouring_surplus(t, solution.network)
        
        """ if continue_balancing_check:
            continue_balancing_check = balance_with_local_storage()
        if continue_balancing_check:
            balance_with_neighbouring_storage()
            continue_balancing_check = balance_with_local_flexible()
        if continue_balancing_check:
            transmit_surplus_flag = balance_with_neighbouring_flexible()

        if transmit_surplus_flag:
            transmit_surplus_to_neighbouring_storage()

        energy_balance.update_storage_energy()
        energy_balance.update_flexible_energy()

        if not precharging_allowed:
            continue

        if not perform_precharge and (interval_memory.deficit.sum() > 1e-6):
            perform_precharge = True

        if perform_precharge and (interval_memory.deficit.sum() < 1e-6):
            precharge_storage(t, energy_balance.residual_load)
            perform_precharge = False """
        
        solution.network.record_final_netloads(t)

        # Test post-energy balance and during energy-balance methods of apportioning
        # Before building the precharge functionality
    return None