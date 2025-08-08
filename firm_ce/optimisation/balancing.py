from firm_ce.common.constants import JIT_ENABLED
from firm_ce.system.topology import Network_InstanceType
from firm_ce.system.components import Fleet_InstanceType
from firm_ce.common.exceptions import raise_unknown_balancing_type_error

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
def initialise_interval_full(interval: int,
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
def initialise_interval_simple(interval: int,
                               network: Network_InstanceType,
                               resolution: float,) -> None:
    for node in network.nodes.values():
        node.initialise_netload_t(interval)
        node.flexible_max_t = 0.0 ###### SET THIS PROPERLY
        node.discharge_max_t = 0.0 ###### SET THIS PROPERLY
        node.charge_max_t = 0.0 ###### SET THIS PROPERLY
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
                               ) -> None:    
    for node in network.nodes.values():
        node.dispatch_storage(interval)
    return None

@njit
def balance_with_flexible(interval: int,
                               network: Network_InstanceType,
                               ) -> None:
    for node in network.nodes.values():
        node.dispatch_flexible(interval) 
    return None

@njit
def apportion_storage_powers(nodes_typed_dict, storages_typed_dict, interval: int) -> None:
    for node in nodes_typed_dict.values():
        for storage_order in node.storage_merit_order:
            if (node.storage_power[interval] < 1e-6) and (node.storage_power[interval] > 1e-6):
                break
            apportioned_value = max(
                min(node.storage_power[interval], storages_typed_dict[storage_order].discharge_max_t),
                -storages_typed_dict[storage_order].charge_max_t
            )
            storages_typed_dict[storage_order].dispatch_power[interval] = apportioned_value
            node.storage_power[interval] -= apportioned_value
    return None
    
@njit
def apportion_flexible_powers(nodes_typed_dict, generators_typed_dict, interval: int) -> None:
    for node in nodes_typed_dict.values():
        for generator_order in node.flexible_merit_order:
            if node.flexible_power[interval] < 1e-6:
                break
            apportioned_value = min(node.flexible_power[interval], generators_typed_dict[generator_order].flexible_max_t)
            generators_typed_dict[generator_order].dispatch_power[interval] = apportioned_value
            node.flexible_power[interval] -= apportioned_value
    return None

@njit
def apportion_storage_simple():
    return None

@njit
def apportion_flexible_simple():
    return None

@njit
def balance_for_period(start_t: int, 
                       end_t: int, 
                       precharging_allowed: bool,
                       solution,
                       ) -> None:
    perform_precharge = False
   
    for t in range(start_t, end_t):
        if solution.balancing_type == 'simple':
            initialise_interval_simple(
                t,
                solution.network,       
                solution.static.resolution
            )
        elif solution.balancing_type == 'full':
            initialise_interval_full(
                t,
                solution.network,
                solution.fleet,            
                solution.static.resolution
            )
        else:
            raise_unknown_balancing_type_error()

        if not precharging_allowed:
            solution.network.reset_transmission(t)

        if solution.network.check_remaining_netloads(t, 'deficit'):
            balance_with_transmission(t, solution.network, 'surplus')
            balance_with_storage(t, solution.network) # Local storage

        if solution.network.check_remaining_netloads(t, 'deficit'):
            balance_with_transmission(t, solution.network, 'storage_discharge')
            balance_with_storage(t, solution.network) # Neighbouring and local storage
            balance_with_flexible(t, solution.network) # Local flexible
        
        if solution.network.check_remaining_netloads(t, 'deficit'):
            balance_with_transmission(t, solution.network, 'flexible')
            balance_with_flexible(t, solution.network) # Neighbouring and local flexible

        if solution.network.check_remaining_netloads(t, 'spillage'):
            balance_with_transmission(t, solution.network, 'storage_charge') 
            balance_with_storage(t, solution.network) # Charge neighbouring storage
            balance_with_flexible(t, solution.network) # Is this needed?
        
        solution.network.calculate_spillage_and_deficit(t)

        if solution.balancing_type == 'simple':
            apportion_storage_simple()
            apportion_flexible_simple()
        elif solution.balancing_type == 'full':            
            apportion_storage_powers(solution.network.nodes, solution.fleet.storages, t)
            apportion_flexible_powers(solution.network.nodes, solution.fleet.generators, t)
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