from firm_ce.common.constants import JIT_ENABLED
from firm_ce.system.topology import Network_InstanceType
from firm_ce.system.components import Fleet_InstanceType

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
def initialise_interval(interval: int,
                        network: Network_InstanceType,
                        fleet: Fleet_InstanceType,
                        resolution: float,) -> None:
    for node in network.nodes.values():
        node.initialise_netload_t(interval)
        node.flexible_max_t = 0.0
        node.discharge_max_t = 0.0
        node.charge_max_t = 0.0
    
    for generator in fleet.generators.values():
        if generator.check_unit_type('flexible'):
            generator.set_flexible_max_t(interval, resolution)
    
    for storage in fleet.storages.values():
        storage.set_dispatch_max_t(interval, resolution)
    return None

@njit 
def balance_with_transmission(interval: int,
                              network: Network_InstanceType,
                              transmission_case: str
                              ) -> None:
    network.set_node_fills_and_surpluses(transmission_case, interval)
    network.fill_with_transmitted_surpluses(interval)
    network.update_netloads(interval)
    return None

@njit
def balance_with_storage(interval: int,
                         network: Network_InstanceType,
                         fleet: Fleet_InstanceType,
                         ) -> None:   
    for node in network.nodes.values():
        for storage_order in node.storage_merit_order:
            fleet.storages[storage_order].dispatch(interval)
    return None

@njit
def balance_with_flexible(interval: int,
                          network: Network_InstanceType,
                          fleet: Fleet_InstanceType,
                          ) -> None:
    for node in network.nodes.values():
        for flexible_order in node.flexible_merit_order:
            # if node.netload_t - node.storage_power[interval] > 0: # Could even make dispatch return a bool?
            fleet.generators[flexible_order].dispatch(interval)
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

        if solution.network.check_remaining_netloads(t, 'deficit'):
            balance_with_transmission(t, solution.network, 'surplus')
            balance_with_storage(t, solution.network, solution.fleet) # Local storage

        if solution.network.check_remaining_netloads(t, 'deficit'):
            balance_with_transmission(t, solution.network, 'storage_discharge')
            balance_with_storage(t, solution.network, solution.fleet) # Neighbouring and local storage
            balance_with_flexible(t, solution.network, solution.fleet) # Local flexible
        
        if solution.network.check_remaining_netloads(t, 'deficit'):
            balance_with_transmission(t, solution.network, 'flexible')
            balance_with_flexible(t, solution.network, solution.fleet) # Neighbouring and local flexible

        if solution.network.check_remaining_netloads(t, 'spillage'):
            balance_with_transmission(t, solution.network, 'storage_charge') 
            balance_with_storage(t, solution.network, solution.fleet) # Charge neighbouring storage
            balance_with_flexible(t, solution.network, solution.fleet) # Is this needed?
        
        solution.network.calculate_spillage_and_deficit(t)

        solution.fleet.update_stored_energies(t, solution.static.resolution)
        solution.fleet.update_remaining_flexible_energies(t, solution.static.resolution)

        if not precharging_allowed:
            continue

        if not perform_precharge and solution.network.check_remaining_netloads(t, 'deficit'):
            perform_precharge = True

        """   

        if perform_precharge and (interval_memory.deficit.sum() < 1e-6):
            precharge_storage(t, energy_balance.residual_load)
            perform_precharge = False """
    return None