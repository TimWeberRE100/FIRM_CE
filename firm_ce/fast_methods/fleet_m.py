import numpy as np
from numpy.typing import NDArray
from typing import Union

from firm_ce.system.components import Fleet, Generator_InstanceType, Storage_InstanceType
from firm_ce.common.constants import JIT_ENABLED, FASTMATH
from firm_ce.common.exceptions import (
    raise_static_modification_error,
)
from firm_ce.fast_methods import generator_m, storage_m
from firm_ce.common.typing import TypedDict, int64
from firm_ce.common.jit_overload import njit

@njit(fastmath=FASTMATH)
def create_dynamic_copy(fleet_instance, nodes_typed_dict, lines_typed_dict):
    generators_copy = TypedDict.empty(
        key_type=int64,
        value_type=Generator_InstanceType
    )
    storages_copy = TypedDict.empty(
        key_type=int64,
        value_type=Storage_InstanceType
    )

    for order, generator in fleet_instance.generators.items():
        generators_copy[order] = generator_m.create_dynamic_copy(generator, nodes_typed_dict, lines_typed_dict)

    for order, storage in fleet_instance.storages.items():
        storages_copy[order] = storage_m.create_dynamic_copy(storage, nodes_typed_dict, lines_typed_dict)

    fleet_copy = Fleet(False,
                 generators_copy,
                 storages_copy,)
        
    return fleet_copy

@njit(fastmath=FASTMATH)
def build_capacities(fleet_instance, decision_x, interval_resolutions: NDArray[np.float64]) -> None:
    if fleet_instance.static_instance:
        raise_static_modification_error()
            
    for generator in fleet_instance.generators.values():
        generator_m.build_capacity(generator, decision_x[generator.candidate_x_idx], interval_resolutions)

    for storage in fleet_instance.storages.values():
        storage_m.build_capacity(storage, decision_x[storage.candidate_p_x_idx], "power")
        storage_m.build_capacity(storage, decision_x[storage.candidate_e_x_idx], "energy")
    return None

@njit(fastmath=FASTMATH)    
def allocate_memory(fleet_instance, intervals_count):
    if fleet_instance.static_instance:
        raise_static_modification_error()

    for generator in fleet_instance.generators.values():
        if generator.unit_type == 'flexible':
            generator_m.allocate_memory(generator, intervals_count)

    for storage in fleet_instance.storages.values():
        storage_m.allocate_memory(storage, intervals_count)
        
    return None

@njit(fastmath=FASTMATH)    
def initialise_stored_energies(fleet_instance):
    if fleet_instance.static_instance:
        raise_static_modification_error()
    for storage in fleet_instance.storages.values():
        storage_m.initialise_stored_energy(storage)
    return None

@njit(fastmath=FASTMATH)    
def initialise_annual_limits(fleet_instance, year: int, first_t: int):
    if fleet_instance.static_instance:
        raise_static_modification_error()        
    for generator in fleet_instance.generators.values():            
        generator_m.initialise_annual_limit(generator, year, first_t)        
    return None

@njit(fastmath=FASTMATH)    
def count_generator_unit_type(fleet_instance, unit_type: str) -> int:
    count = 0
    for generator in fleet_instance.generators.values():
        if generator.unit_type == unit_type:
            count+=1
    return count

@njit(fastmath=FASTMATH)    
def update_stored_energies(fleet_instance, interval: int, resolution: float) -> None:
    for storage in fleet_instance.storages.values():
        storage_m.update_stored_energy(storage, interval, resolution)
    return None

@njit(fastmath=FASTMATH)    
def update_remaining_flexible_energies(fleet_instance, interval: int, resolution: float) -> None:
    for generator in fleet_instance.generators.values():
        if not generator_m.check_unit_type(generator, 'flexible'):
            continue
        generator_m.update_remaining_energy(generator, interval, resolution)
    return None

@njit(fastmath=FASTMATH)    
def calculate_lt_generations(fleet_instance, interval_resolutions: NDArray[np.float64]) -> None:
    for generator in fleet_instance.generators.values():
        if generator_m.check_unit_type(generator, 'flexible'):
            generator_m.calculate_lt_generation(generator, interval_resolutions)

    for storage in fleet_instance.storages.values():
        storage_m.calculate_lt_discharge(storage, interval_resolutions)
    return None