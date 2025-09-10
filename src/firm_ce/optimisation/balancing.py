from firm_ce.common.constants import FASTMATH
from firm_ce.common.jit_overload import njit
from firm_ce.common.typing import boolean, float64, int64, unicode_type
from firm_ce.fast_methods import (
    fleet_m,
    generator_m,
    network_m,
    node_m,
    static_m,
    storage_m,
)
from firm_ce.system.components import Fleet_InstanceType
from firm_ce.system.topology import Network_InstanceType


@njit(fastmath=FASTMATH)
def initialise_interval(
    interval: int64,
    network: Network_InstanceType,
    fleet: Fleet_InstanceType,
    resolution: float64,
    forward_time_flag: boolean,
) -> None:
    for node in network.nodes.values():
        node_m.initialise_netload_t(node, interval)
        node_m.reset_dispatch_max_t(node)

        for idx, flexible_order in enumerate(node.flexible_merit_order):
            generator_m.set_flexible_max_t(
                fleet.generators[flexible_order], interval, resolution, idx, forward_time_flag
            )
        for idx, storage_order in enumerate(node.storage_merit_order):
            storage_m.set_dispatch_max_t(fleet.storages[storage_order], interval, resolution, idx, forward_time_flag)
    return None


@njit(fastmath=FASTMATH)
def balance_with_transmission(
    interval: int64, network: Network_InstanceType, transmission_case: unicode_type, precharging_flag: boolean
) -> None:
    network_m.set_node_fills_and_surpluses(network, transmission_case, interval)
    network_m.fill_with_transmitted_surpluses(network, interval)
    network_m.update_netloads(network, interval, precharging_flag)

    if precharging_flag:
        network_m.update_imports_exports_temp(network, interval)
    return None


@njit(fastmath=FASTMATH)
def balance_with_storage(
    interval: int64,
    network: Network_InstanceType,
    fleet: Fleet_InstanceType,
) -> None:
    for node in network.nodes.values():
        if not node_m.check_remaining_netload(node, interval, "both"):
            continue
        node.storage_power[interval] = 0
        for idx, storage_order in enumerate(node.storage_merit_order):
            storage_m.dispatch(fleet.storages[storage_order], interval, idx)
    return None


@njit(fastmath=FASTMATH)
def balance_with_flexible(
    interval: int64,
    network: Network_InstanceType,
    fleet: Fleet_InstanceType,
) -> None:
    for node in network.nodes.values():
        if not node_m.check_remaining_netload(node, interval, "deficit"):
            continue
        node.flexible_power[interval] = 0
        for idx, flexible_order in enumerate(node.flexible_merit_order):
            generator_m.dispatch(fleet.generators[flexible_order], interval, idx)
    return None


@njit(fastmath=FASTMATH)
def energy_balance_for_interval(
    solution, interval: int64, forward_time_flag: boolean  # Solution_InstanceType
) -> int64:
    initialise_interval(
        interval,
        solution.network,
        solution.fleet,
        solution.static.interval_resolutions[interval],
        forward_time_flag,
    )

    if network_m.check_remaining_netloads(solution.network, interval, "deficit"):
        balance_with_transmission(interval, solution.network, "surplus", False)
        balance_with_storage(interval, solution.network, solution.fleet)  # Local storage

    if network_m.check_remaining_netloads(solution.network, interval, "deficit"):
        balance_with_transmission(interval, solution.network, "storage_discharge", False)
        balance_with_storage(interval, solution.network, solution.fleet)  # Neighbouring and local storage
        balance_with_flexible(interval, solution.network, solution.fleet)  # Local flexible

    if network_m.check_remaining_netloads(solution.network, interval, "deficit"):
        balance_with_transmission(interval, solution.network, "flexible", False)
        balance_with_flexible(interval, solution.network, solution.fleet)  # Neighbouring and local flexible

    if network_m.check_remaining_netloads(solution.network, interval, "spillage"):
        balance_with_transmission(interval, solution.network, "storage_charge", False)
        balance_with_storage(interval, solution.network, solution.fleet)  # Charge neighbouring storage

    return None


@njit(fastmath=FASTMATH)
def balance_for_period(
    start_t: int64,
    end_t: int64,
    precharging_allowed: boolean,
    solution,  # Solution_InstanceType
    year: int64,
) -> None:
    perform_precharge = False

    for t in range(start_t, end_t):
        energy_balance_for_interval(solution, t, True)

        network_m.calculate_spillage_and_deficit(solution.network, t)

        fleet_m.update_stored_energies(solution.fleet, t, solution.static.interval_resolutions[t], True)
        fleet_m.update_remaining_flexible_energies(
            solution.fleet, t, solution.static.interval_resolutions[t], True, False
        )

        if not precharging_allowed:
            continue

        if not perform_precharge and network_m.check_remaining_netloads(solution.network, t, "deficit"):
            perform_precharge = True

        if perform_precharge and (
            not network_m.check_remaining_netloads(solution.network, t, "deficit") or t == end_t - 1
        ):  # and t<500: # DEBUG 500
            precharge_storage(solution, t, year)
            perform_precharge = False
    return None


@njit(fastmath=FASTMATH)
def determine_precharge_energies(interval: int64, solution, year: int64) -> int64:  # Solution_InstanceType
    fleet_m.initialise_deficit_block(solution.fleet, interval)
    previous_year_flag = False
    precharging_year = year
    first_t, _ = static_m.get_year_t_boundaries(solution.static, precharging_year)

    while True:
        if interval == first_t:
            previous_year_flag = True
            precharging_year -= 1
            first_t, _ = static_m.get_year_t_boundaries(solution.static, precharging_year)

        interval -= 1

        if interval < 0:
            return 0

        network_m.reset_transmission(solution.network, interval)
        network_m.reset_dispatch(solution.network, interval)
        fleet_m.reset_dispatch(solution.fleet, interval)

        energy_balance_for_interval(solution, interval, False)

        fleet_m.update_stored_energies(solution.fleet, interval, solution.static.interval_resolutions[interval], False)
        fleet_m.update_remaining_flexible_energies(
            solution.fleet, interval, solution.static.interval_resolutions[interval], False, previous_year_flag
        )
        fleet_m.update_deficit_block(solution.fleet)

        if network_m.check_precharging_end(solution.network, interval):
            fleet_m.assign_precharging_values(
                solution.fleet, interval, solution.static.interval_resolutions[interval], year
            )
            return interval

        previous_year_flag = False


@njit(fastmath=FASTMATH)
def initialise_precharging_interval(
    interval: int64, network: Network_InstanceType, fleet: Fleet_InstanceType, resolution: float64
) -> None:
    fleet_m.update_precharging_flags(fleet, interval)

    for node in network.nodes.values():
        node_m.set_imports_exports_temp(node, interval)
        node_m.update_netload_t(node, interval, True)
        node_m.reset_dispatch_max_t(node)

        for idx, storage_order in enumerate(node.storage_merit_order):
            storage_m.set_precharging_max_t(fleet.storages[storage_order], interval, resolution, idx)
        for idx, flexible_order in enumerate(node.flexible_merit_order):
            generator_m.set_precharging_max_t(fleet.generators[flexible_order], interval, resolution, idx)
    return None


@njit(fastmath=FASTMATH)
def perform_local_surplus_transfers(
    interval: int64, network: Network_InstanceType, fleet: Fleet_InstanceType, resolution: float64
) -> None:
    for node in network.nodes.values():
        node.existing_surplus = -min(0, node.netload_t)
        if node.existing_surplus < 1e-6:
            continue

        for idx, storage_order in enumerate(node.storage_merit_order[::-1]):
            if not fleet.storages[storage_order].precharge_flag:
                continue

            dispatch_power_update = min(max(-node.existing_surplus, -fleet.storages[storage_order].charge_max_t), 0.0)
            storage_m.update_precharge_dispatch(
                fleet.storages[storage_order], interval, resolution, dispatch_power_update, True, idx
            )
            node.existing_surplus += dispatch_power_update
    return None


@njit(fastmath=FASTMATH)
def perform_transmitted_surplus_transfers(
    interval: int64, network: Network_InstanceType, fleet: Fleet_InstanceType, resolution: float64
) -> None:
    if not network_m.check_existing_surplus(network):
        return None

    balance_with_transmission(interval, network, "precharging_surplus", True)

    for node in network.nodes.values():
        for idx, storage_order in enumerate(node.storage_merit_order[::-1]):
            if not fleet.storages[storage_order].precharge_flag:
                continue

            dispatch_power_update = min(
                max(node.imports_exports_update, -fleet.storages[storage_order].charge_max_t), 0.0
            )
            storage_m.update_precharge_dispatch(
                fleet.storages[storage_order], interval, resolution, dispatch_power_update, True, idx
            )
            node.imports_exports_update -= dispatch_power_update
    return None


@njit(fastmath=FASTMATH)
def perform_intranode_interstorage_transfers(
    interval: int64, network: Network_InstanceType, fleet: Fleet_InstanceType, resolution: float64
) -> None:
    for node in network.nodes.values():
        intranode_transfer_power = min(node.precharge_surplus, node.precharge_fill)
        intranode_trickle = intranode_transfer_power
        intranode_precharge = -intranode_transfer_power

        for idx, storage_order in enumerate(node.storage_merit_order[::-1]):
            if not fleet.storages[storage_order].precharge_flag and not fleet.storages[storage_order].trickling_flag:
                continue

            if fleet.storages[storage_order].trickling_flag:
                dispatch_power_update = max(min(intranode_trickle, fleet.storages[storage_order].discharge_max_t), 0.0)
                storage_m.update_precharge_dispatch(
                    fleet.storages[storage_order], interval, resolution, dispatch_power_update, False, idx
                )
                intranode_trickle -= dispatch_power_update

            if fleet.storages[storage_order].precharge_flag:
                dispatch_power_update = min(max(intranode_precharge, -fleet.storages[storage_order].charge_max_t), 0.0)
                storage_m.update_precharge_dispatch(
                    fleet.storages[storage_order], interval, resolution, dispatch_power_update, True, idx
                )
                intranode_precharge -= dispatch_power_update
    return None


@njit(fastmath=FASTMATH)
def perform_internode_interstorage_transfers(
    interval: int64, network: Network_InstanceType, fleet: Fleet_InstanceType, resolution: float64
) -> None:
    if not (network_m.check_precharge_fill(network) and network_m.check_precharge_surplus(network)):
        return None

    balance_with_transmission(interval, network, "precharging_transfers", True)

    for node in network.nodes.values():

        for idx, storage_order in enumerate(node.storage_merit_order[::-1]):
            if not fleet.storages[storage_order].precharge_flag and not fleet.storages[storage_order].trickling_flag:
                continue

            if fleet.storages[storage_order].trickling_flag:
                dispatch_power_update = max(
                    min(node.imports_exports_update, fleet.storages[storage_order].discharge_max_t), 0.0
                )
                storage_m.update_precharge_dispatch(
                    fleet.storages[storage_order], interval, resolution, dispatch_power_update, False, idx
                )
                node.imports_exports_update -= dispatch_power_update

            if fleet.storages[storage_order].precharge_flag:
                dispatch_power_update = min(
                    max(node.imports_exports_update, -fleet.storages[storage_order].charge_max_t), 0.0
                )
                storage_m.update_precharge_dispatch(
                    fleet.storages[storage_order], interval, resolution, dispatch_power_update, True, idx
                )
                node.imports_exports_update -= dispatch_power_update
    return None


@njit(fastmath=FASTMATH)
def perform_intranode_flexible_transfers(
    interval: int64, network: Network_InstanceType, fleet: Fleet_InstanceType, resolution: float64
) -> None:
    for node in network.nodes.values():
        intranode_transfer_power = min(node.precharge_surplus, node.precharge_fill)
        intranode_trickle = intranode_transfer_power
        intranode_precharge = -intranode_transfer_power

        for idx, flexible_order in enumerate(node.flexible_merit_order):
            if not fleet.generators[flexible_order].trickling_flag:
                continue

            dispatch_power_update = max(min(intranode_trickle, fleet.generators[flexible_order].flexible_max_t), 0.0)
            generator_m.update_precharge_dispatch(
                fleet.generators[flexible_order], interval, resolution, dispatch_power_update, idx
            )
            intranode_trickle -= dispatch_power_update

        for idx, storage_order in enumerate(node.storage_merit_order):
            if not fleet.storages[storage_order].precharge_flag:
                continue

            dispatch_power_update = min(max(intranode_precharge, -fleet.storages[storage_order].charge_max_t), 0.0)
            storage_m.update_precharge_dispatch(
                fleet.storages[storage_order], interval, resolution, dispatch_power_update, True, idx
            )
            intranode_precharge -= dispatch_power_update
    return None


@njit(fastmath=FASTMATH)
def perform_internode_flexible_transfers(
    interval: int64, network: Network_InstanceType, fleet: Fleet_InstanceType, resolution: float64
) -> None:
    if not (network_m.check_precharge_fill(network) and network_m.check_precharge_surplus(network)):
        return None

    balance_with_transmission(interval, network, "precharging_transfers", True)

    for node in network.nodes.values():
        for idx, flexible_order in enumerate(node.flexible_merit_order):
            if not fleet.generators[flexible_order].trickling_flag:
                continue
            dispatch_power_update = max(
                min(node.imports_exports_update, fleet.generators[flexible_order].flexible_max_t), 0.0
            )
            generator_m.update_precharge_dispatch(
                fleet.generators[flexible_order], interval, resolution, dispatch_power_update, idx
            )
            node.imports_exports_update -= dispatch_power_update

        for idx, storage_order in enumerate(node.storage_merit_order[::-1]):
            if not fleet.storages[storage_order].precharge_flag:
                continue
            if fleet.storages[storage_order].precharge_flag:
                dispatch_power_update = min(
                    max(node.imports_exports_update, -fleet.storages[storage_order].charge_max_t), 0.0
                )
                storage_m.update_precharge_dispatch(
                    fleet.storages[storage_order], interval, resolution, dispatch_power_update, True, idx
                )
                node.imports_exports_update -= dispatch_power_update
    return None


@njit(fastmath=FASTMATH)
def perform_flexible_precharging(solution, interval: int64) -> None:  # Solution_InstanceType
    network_m.set_flexible_precharge_fills_and_surpluses(solution.network)
    for node in solution.network.nodes.values():
        node_m.set_imports_exports_temp(node, interval)

    perform_intranode_flexible_transfers(
        interval,
        solution.network,
        solution.fleet,
        solution.static.interval_resolutions[interval],
    )

    perform_internode_flexible_transfers(
        interval,
        solution.network,
        solution.fleet,
        solution.static.interval_resolutions[interval],
    )

    return None


@njit(fastmath=FASTMATH)
def determine_precharge_powers(interval: int64, solution, year: int64) -> int64:  # Solution_InstanceType
    first_interval_precharge = 0
    fleet_m.initialise_precharging_flags(solution.fleet, interval)
    precharging_year = year
    first_t, _ = static_m.get_year_t_boundaries(solution.static, precharging_year)

    while True:
        if interval == first_t:
            precharging_year -= 1
            first_t, _ = static_m.get_year_t_boundaries(solution.static, precharging_year)
            fleet_m.reset_flexible_reserves(solution.fleet)

        interval -= 1

        if interval < 0:
            first_interval_precharge = 0
            break

        initialise_precharging_interval(
            interval,
            solution.network,
            solution.fleet,
            solution.static.interval_resolutions[interval],
        )
        network_m.set_storage_precharge_fills_and_surpluses(solution.network)

        perform_local_surplus_transfers(
            interval,
            solution.network,
            solution.fleet,
            solution.static.interval_resolutions[interval],
        )

        perform_transmitted_surplus_transfers(
            interval,
            solution.network,
            solution.fleet,
            solution.static.interval_resolutions[interval],
        )

        perform_intranode_interstorage_transfers(
            interval,
            solution.network,
            solution.fleet,
            solution.static.interval_resolutions[interval],
        )

        fleet_m.update_precharging_flags(solution.fleet, interval)

        perform_internode_interstorage_transfers(
            interval,
            solution.network,
            solution.fleet,
            solution.static.interval_resolutions[interval],
        )

        fleet_m.update_precharging_flags(solution.fleet, interval)

        if fleet_m.check_precharge_remaining(solution.fleet):
            perform_flexible_precharging(solution, interval)

        if (not fleet_m.check_precharge_remaining(solution.fleet)) or (
            not fleet_m.check_trickling_remaining(solution.fleet)
        ):
            first_interval_precharge = interval
            break

    return first_interval_precharge


@njit(fastmath=FASTMATH)
def perform_fill_adjustment(interval: int64, network: Network_InstanceType, fleet: Fleet_InstanceType) -> None:
    for node in network.nodes.values():
        for storage_order in node.storage_merit_order:
            dispatch_power_update = max(
                min(node.fill, fleet.storages[storage_order].remaining_discharge_max_t),
                -fleet.storages[storage_order].remaining_charge_max_t,
            )
            fleet.storages[storage_order].dispatch_power[interval] += dispatch_power_update
            node.storage_power[interval] += dispatch_power_update
            node.fill -= dispatch_power_update
    return None


@njit(fastmath=FASTMATH)
def update_precharge_stored_energy(
    first_interval_precharge: int64, interval_after_deficit_block: int64, solution  # Solution_InstanceType
) -> None:
    for interval in range(first_interval_precharge, interval_after_deficit_block):
        initialise_interval(
            interval,
            solution.network,
            solution.fleet,
            solution.static.interval_resolutions[interval],
            True,
        )

        infeasible_flag = fleet_m.determine_feasible_storage_dispatch(solution.fleet, interval)
        if infeasible_flag:
            network_m.reset_transmission(solution.network, interval)

            balance_with_transmission(interval, solution.network, "precharging_adjust_storage", False)
            balance_with_flexible(interval, solution.network, solution.fleet)  # Local flexible

            if network_m.check_remaining_netloads(solution.network, interval, "deficit"):
                balance_with_transmission(interval, solution.network, "flexible", False)
                balance_with_flexible(interval, solution.network, solution.fleet)  # Neighbouring and local flexible
        else:
            infeasible_flag = fleet_m.determine_feasible_flexible_dispatch(solution.fleet, interval)
            if infeasible_flag:
                network_m.reset_transmission(solution.network, interval)
                fleet_m.calculate_available_storage_dispatch(solution.fleet, interval)

                balance_with_transmission(interval, solution.network, "precharging_adjust_surplus", False)
                balance_with_transmission(interval, solution.network, "precharging_adjust_flexible", False)

                perform_fill_adjustment(
                    interval,
                    solution.network,
                    solution.fleet,
                )
            else:
                network_m.update_netloads(solution.network, interval, False)

        network_m.calculate_spillage_and_deficit(solution.network, interval)

        fleet_m.update_stored_energies(solution.fleet, interval, solution.static.interval_resolutions[interval], True)
        fleet_m.update_remaining_flexible_energies(
            solution.fleet, interval, solution.static.interval_resolutions[interval], True, False
        )
    return None


@njit(fastmath=FASTMATH)
def precharge_storage(solution, interval_after_deficit_block: int64, year: int64) -> None:  # Solution_InstanceType
    first_interval_deficit_block = determine_precharge_energies(interval_after_deficit_block, solution, year)

    first_interval_precharge = determine_precharge_powers(first_interval_deficit_block, solution, year)

    update_precharge_stored_energy(first_interval_precharge, interval_after_deficit_block + 1, solution)
    return None
