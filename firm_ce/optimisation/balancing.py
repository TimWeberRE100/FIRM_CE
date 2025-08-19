import numpy as np

from firm_ce.common.constants import JIT_ENABLED, FASTMATH
from firm_ce.fast_methods import (
    node_m, network_m,
    generator_m, storage_m, fleet_m,
)
from firm_ce.system.topology import Network_InstanceType
from firm_ce.system.components import Fleet_InstanceType
from firm_ce.common.jit_overload import njit

@njit(fastmath=FASTMATH) 
def initialise_interval(interval: int,
                        network: Network_InstanceType,
                        fleet: Fleet_InstanceType,
                        resolution: float,) -> None:
    for node in network.nodes.values():
        node_m.initialise_netload_t(node, interval)
        if len(node.flexible_merit_order) > 0:
            node.flexible_max_t = np.zeros(len(node.flexible_merit_order), dtype=np.float64)
        else:
            node.flexible_max_t = np.zeros(1, dtype=np.float64)
        if len(node.storage_merit_order) > 0:
            node.discharge_max_t = np.zeros(len(node.storage_merit_order), dtype=np.float64)
            node.charge_max_t = np.zeros(len(node.storage_merit_order), dtype=np.float64)
        else:
            node.discharge_max_t = np.zeros(1, dtype=np.float64)
            node.charge_max_t = np.zeros(1, dtype=np.float64)

        for idx, flexible_order in enumerate(node.flexible_merit_order):
            generator_m.set_flexible_max_t(fleet.generators[flexible_order], interval, resolution, idx)
        for idx, storage_order in enumerate(node.storage_merit_order):
            storage_m.set_dispatch_max_t(fleet.storages[storage_order], interval, resolution, idx)
    return None

@njit(fastmath=FASTMATH)
def balance_with_transmission(interval: int,
                              network: Network_InstanceType,
                              transmission_case: str
                              ) -> None:
    network_m.set_node_fills_and_surpluses(network, transmission_case, interval)
    network_m.fill_with_transmitted_surpluses(network, interval)
    network_m.update_netloads(network, interval)
    return None

@njit(fastmath=FASTMATH)
def balance_with_storage(interval: int,
                         network: Network_InstanceType,
                         fleet: Fleet_InstanceType,
                         ) -> None:   
    for node in network.nodes.values():
        if not node_m.check_remaining_netload(node, interval, 'both'):
            continue 
        node.storage_power[interval] = 0
        for idx, storage_order in enumerate(node.storage_merit_order):
            storage_m.dispatch(fleet.storages[storage_order], interval, idx)
    return None

@njit(fastmath=FASTMATH)
def balance_with_flexible(interval: int,
                          network: Network_InstanceType,
                          fleet: Fleet_InstanceType,
                          ) -> None:
    for node in network.nodes.values():
        if not node_m.check_remaining_netload(node, interval, 'deficit'):
            continue
        node.flexible_power[interval] = 0
        for idx, flexible_order in enumerate(node.flexible_merit_order):
            generator_m.dispatch(fleet.generators[flexible_order], interval, idx)
    return None

@njit(fastmath=FASTMATH)
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
            network_m.reset_transmission(solution.network, t)

        if network_m.check_remaining_netloads(solution.network, t, 'deficit'):
            balance_with_transmission(t, solution.network, 'surplus')
            balance_with_storage(t, solution.network, solution.fleet) # Local storage

        if network_m.check_remaining_netloads(solution.network, t, 'deficit'):
            balance_with_transmission(t, solution.network, 'storage_discharge')
            balance_with_storage(t, solution.network, solution.fleet) # Neighbouring and local storage
            balance_with_flexible(t, solution.network, solution.fleet) # Local flexible
        
        if network_m.check_remaining_netloads(solution.network, t, 'deficit'):
            balance_with_transmission(t, solution.network, 'flexible')
            balance_with_flexible(t, solution.network, solution.fleet) # Neighbouring and local flexible

        if network_m.check_remaining_netloads(solution.network, t, 'spillage'):
            balance_with_transmission(t, solution.network, 'storage_charge') 
            balance_with_storage(t, solution.network, solution.fleet) # Charge neighbouring storage           

        network_m.calculate_spillage_and_deficit(solution.network, t)

        fleet_m.update_stored_energies(solution.fleet, t, solution.static.resolution)
        fleet_m.update_remaining_flexible_energies(solution.fleet, t, solution.static.resolution)

        if not precharging_allowed:
            continue

        if not perform_precharge and network_m.check_remaining_netloads(solution.network, t, 'deficit'):
            perform_precharge = True

        """   

        if perform_precharge and (interval_memory.deficit.sum() < 1e-6):
            precharge_storage(t, energy_balance.residual_load)
            perform_precharge = False """
    return None
