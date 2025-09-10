from firm_ce.common.constants import FASTMATH
from firm_ce.common.exceptions import raise_static_modification_error
from firm_ce.common.jit_overload import njit
from firm_ce.common.typing import DictType, TypedDict, boolean, float64, int64, unicode_type
from firm_ce.system.components import Fleet, Fleet_InstanceType, Generator_InstanceType, Storage_InstanceType
from firm_ce.system.topology import Line_InstanceType, Node_InstanceType
from firm_ce.fast_methods import generator_m, storage_m


@njit(fastmath=FASTMATH)
def create_dynamic_copy(
    fleet_instance: Fleet_InstanceType,
    nodes_typed_dict: DictType(int64, Node_InstanceType),
    lines_typed_dict: DictType(int64, Line_InstanceType),
) -> Fleet_InstanceType:
    generators_copy = TypedDict.empty(key_type=int64, value_type=Generator_InstanceType)
    storages_copy = TypedDict.empty(key_type=int64, value_type=Storage_InstanceType)

    for order, generator in fleet_instance.generators.items():
        generators_copy[order] = generator_m.create_dynamic_copy(generator, nodes_typed_dict, lines_typed_dict)

    for order, storage in fleet_instance.storages.items():
        storages_copy[order] = storage_m.create_dynamic_copy(storage, nodes_typed_dict, lines_typed_dict)

    fleet_copy = Fleet(
        False,
        generators_copy,
        storages_copy,
    )

    return fleet_copy


@njit(fastmath=FASTMATH)
def build_capacities(
    fleet_instance: Fleet_InstanceType, decision_x: float64[:], interval_resolutions: float64[:]
) -> None:
    if fleet_instance.static_instance:
        raise_static_modification_error()

    for generator in fleet_instance.generators.values():
        generator_m.build_capacity(generator, decision_x[generator.candidate_x_idx], interval_resolutions)

    for storage in fleet_instance.storages.values():
        storage_m.build_capacity(storage, decision_x[storage.candidate_p_x_idx], "power")
        storage_m.build_capacity(storage, decision_x[storage.candidate_e_x_idx], "energy")
    return None


@njit(fastmath=FASTMATH)
def allocate_memory(fleet_instance: Fleet_InstanceType, intervals_count: int64) -> None:
    if fleet_instance.static_instance:
        raise_static_modification_error()

    for generator in fleet_instance.generators.values():
        if generator.unit_type == "flexible":
            generator_m.allocate_memory(generator, intervals_count)

    for storage in fleet_instance.storages.values():
        storage_m.allocate_memory(storage, intervals_count)

    return None


@njit(fastmath=FASTMATH)
def initialise_stored_energies(fleet_instance: Fleet_InstanceType) -> None:
    if fleet_instance.static_instance:
        raise_static_modification_error()
    for storage in fleet_instance.storages.values():
        storage_m.initialise_stored_energy(storage)
    return None


@njit(fastmath=FASTMATH)
def initialise_annual_limits(fleet_instance: Fleet_InstanceType, year: int64, first_t: int64) -> None:
    if fleet_instance.static_instance:
        raise_static_modification_error()
    for generator in fleet_instance.generators.values():
        generator_m.initialise_annual_limit(generator, year, first_t)
    return None


@njit(fastmath=FASTMATH)
def count_generator_unit_type(fleet_instance: Fleet_InstanceType, unit_type: unicode_type) -> int64:
    count = 0
    for generator in fleet_instance.generators.values():
        if generator.unit_type == unit_type:
            count += 1
    return count


@njit(fastmath=FASTMATH)
def update_stored_energies(
    fleet_instance: Fleet_InstanceType, interval: int64, resolution: float64, forward_time_flag: boolean
) -> None:
    for storage in fleet_instance.storages.values():
        storage_m.update_stored_energy(storage, interval, resolution, forward_time_flag)
    return None


@njit(fastmath=FASTMATH)
def update_remaining_flexible_energies(
    fleet_instance: Fleet_InstanceType,
    interval: int64,
    resolution: float64,
    forward_time_flag: boolean,
    previous_year_flag: boolean,
) -> None:
    for generator in fleet_instance.generators.values():
        if not generator_m.check_unit_type(generator, "flexible"):
            continue
        generator_m.update_remaining_energy(generator, interval, resolution, forward_time_flag, previous_year_flag)
    return None


@njit(fastmath=FASTMATH)
def calculate_lt_generations(fleet_instance: Fleet_InstanceType, interval_resolutions: float64[:]) -> None:
    for generator in fleet_instance.generators.values():
        if generator_m.check_unit_type(generator, "flexible"):
            generator_m.calculate_lt_generation(generator, interval_resolutions)

    for storage in fleet_instance.storages.values():
        storage_m.calculate_lt_discharge(storage, interval_resolutions)
    return None


@njit(fastmath=FASTMATH)
def initialise_deficit_block(fleet_instance: Fleet_InstanceType, interval_after_deficit_block: int64) -> None:
    for storage in fleet_instance.storages.values():
        storage_m.initialise_deficit_block(storage, interval_after_deficit_block)

    for generator in fleet_instance.generators.values():
        if generator_m.check_unit_type(generator, "flexible"):
            generator_m.initialise_deficit_block(generator, interval_after_deficit_block)

    return None


@njit(fastmath=FASTMATH)
def reset_dispatch(fleet_instance: Fleet_InstanceType, interval: int64) -> None:
    for storage in fleet_instance.storages.values():
        storage.dispatch_power[interval] = 0.0
    for generator in fleet_instance.generators.values():
        if generator_m.check_unit_type(generator, "flexible"):
            generator.dispatch_power[interval] = 0.0
    return None


@njit(fastmath=FASTMATH)
def update_deficit_block(fleet_instance: Fleet_InstanceType) -> None:
    for storage in fleet_instance.storages.values():
        storage_m.update_deficit_block_bounds(storage, storage.stored_energy_temp_reverse)

    for generator in fleet_instance.generators.values():
        if generator_m.check_unit_type(generator, "flexible"):
            generator_m.update_deficit_block_bounds(generator, generator.remaining_energy_temp_reverse)
    return None


@njit(fastmath=FASTMATH)
def assign_precharging_values(
    fleet_instance: Fleet_InstanceType, interval: int64, resolution: float64, year: int64
) -> None:
    for generator in fleet_instance.generators.values():
        if generator_m.check_unit_type(generator, "flexible"):
            generator.remaining_energy_temp_forward = (
                generator.remaining_energy[interval - 1] - generator.dispatch_power[interval] * resolution
            )
            generator.remaining_energy_temp_forward = min(
                max(generator.remaining_energy_temp_forward, 0.0), generator_m.get_annual_limit(generator, year)
            )
            generator_m.update_deficit_block_bounds(generator, generator.remaining_energy_temp_forward)
            generator_m.assign_trickling_reserves(generator)

    for storage in fleet_instance.storages.values():
        # After reverse charging, the stored energy is discontinuous in the forward and reverse directions
        storage.stored_energy_temp_forward = (
            storage.stored_energy[interval - 1]
            - max(storage.dispatch_power[interval], 0) / storage.discharge_efficiency * resolution
            - min(storage.dispatch_power[interval], 0) * storage.charge_efficiency * resolution
        )
        storage.stored_energy_temp_forward = min(max(storage.stored_energy_temp_forward, 0.0), storage.energy_capacity)
        storage_m.update_deficit_block_bounds(storage, storage.stored_energy_temp_forward)
        storage_m.assign_precharging_reserves(storage)
    return None


@njit(fastmath=FASTMATH)
def initialise_precharging_flags(fleet_instance: Fleet_InstanceType, interval: int64) -> None:
    for storage in fleet_instance.storages.values():
        storage_m.initialise_precharging_flags(storage, interval)

    for generator in fleet_instance.generators.values():
        if generator_m.check_unit_type(generator, "flexible"):
            generator_m.initialise_precharging_flags(generator, interval)
    return None


@njit(fastmath=FASTMATH)
def update_precharging_flags(fleet_instance: Fleet_InstanceType, interval: int64) -> None:
    for storage in fleet_instance.storages.values():
        storage_m.update_precharging_flags(storage, interval)
    for generator in fleet_instance.generators.values():
        if generator_m.check_unit_type(generator, "flexible"):
            generator_m.update_precharging_flags(generator, interval)
    return None


@njit(fastmath=FASTMATH)
def check_precharge_remaining(fleet_instance: Fleet_InstanceType) -> boolean:
    for storage in fleet_instance.storages.values():
        if storage.precharge_flag:
            return True
    return False


@njit(fastmath=FASTMATH)
def check_trickling_remaining(fleet_instance: Fleet_InstanceType) -> boolean:
    for storage in fleet_instance.storages.values():
        if storage.trickling_flag:
            return True
    for generator in fleet_instance.generators.values():
        if not generator_m.check_unit_type(generator, "flexible"):
            continue
        if generator.trickling_flag:
            return True
    return False


@njit(fastmath=FASTMATH)
def determine_feasible_storage_dispatch(fleet_instance: Fleet_InstanceType, interval: int64) -> boolean:
    infeasible_flag = False
    for storage in fleet_instance.storages.values():
        original_dispatch_power = storage.dispatch_power[interval]
        storage.dispatch_power[interval] = max(min(original_dispatch_power, storage.discharge_max_t), 0.0) + min(
            max(original_dispatch_power, -storage.charge_max_t), 0.0
        )
        dispatch_power_adjustment = original_dispatch_power - storage.dispatch_power[interval]
        if abs(dispatch_power_adjustment) > 1e-6:
            storage.node.storage_power[interval] -= dispatch_power_adjustment
            infeasible_flag = True
    return infeasible_flag


@njit(fastmath=FASTMATH)
def determine_feasible_flexible_dispatch(fleet_instance: Fleet_InstanceType, interval: int64) -> boolean:
    infeasible_flag = False
    for generator in fleet_instance.generators.values():
        if not generator_m.check_unit_type(generator, "flexible"):
            continue
        original_dispatch_power = generator.dispatch_power[interval]
        generator.dispatch_power[interval] = min(original_dispatch_power, generator.flexible_max_t)
        dispatch_power_adjustment = original_dispatch_power - generator.dispatch_power[interval]
        if abs(dispatch_power_adjustment) > 1e-6:
            generator.node.flexible_power[interval] -= dispatch_power_adjustment
            infeasible_flag = True
    return infeasible_flag


@njit(fastmath=FASTMATH)
def calculate_available_storage_dispatch(fleet_instance: Fleet_InstanceType, interval: int64) -> None:
    for storage in fleet_instance.storages.values():
        storage_m.calculate_available_dispatch(storage, interval)


@njit(fastmath=FASTMATH)
def reset_flexible_reserves(fleet_instance: Fleet_InstanceType) -> None:
    for generator in fleet_instance.generators.values():
        if not generator_m.check_unit_type(generator, "flexible"):
            continue
        generator.trickling_reserves = 0
    return None
