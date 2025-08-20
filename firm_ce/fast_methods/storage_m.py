import numpy as np
from numpy.typing import NDArray

from firm_ce.common.constants import JIT_ENABLED, FASTMATH
from firm_ce.system.components import Storage
from firm_ce.common.exceptions import (
    raise_static_modification_error,
)
from firm_ce.fast_methods import node_m, ltcosts_m
from firm_ce.common.jit_overload import njit

@njit(fastmath=FASTMATH)
def create_dynamic_copy(storage_instance, nodes_typed_dict, lines_typed_dict):
    node_copy = nodes_typed_dict[storage_instance.node.order]
    line_copy = lines_typed_dict[storage_instance.line.order] 

    storage_copy = Storage(
        False,
        storage_instance.id,
        storage_instance.order,
        storage_instance.name,
        storage_instance.power_capacity,
        storage_instance.energy_capacity,
        storage_instance.duration,
        storage_instance.charge_efficiency,
        storage_instance.discharge_efficiency,
        storage_instance.max_build_p,
        storage_instance.max_build_e,
        storage_instance.min_build_p,
        storage_instance.min_build_e,
        storage_instance.unit_type,
        storage_instance.near_optimum_check,
        node_copy,
        line_copy,
        storage_instance.group, 
        storage_instance.cost, # This remains static
    )

    storage_copy.candidate_p_x_idx = storage_instance.candidate_p_x_idx
    storage_copy.candidate_e_x_idx = storage_instance.candidate_e_x_idx    
        
    return storage_copy

@njit(fastmath=FASTMATH)
def build_capacity(storage_instance, new_build_capacity, capacity_type):
    if storage_instance.static_instance:
        raise_static_modification_error()
    if capacity_type == "power":
        storage_instance.power_capacity += new_build_capacity
        storage_instance.new_build_p += new_build_capacity
        storage_instance.line.capacity += new_build_capacity

        if storage_instance.duration > 0:
            storage_instance.energy_capacity += new_build_capacity * storage_instance.duration
            storage_instance.new_build_e += new_build_capacity * storage_instance.duration
            
    if capacity_type == "energy":
        if storage_instance.duration == 0:
            storage_instance.energy_capacity += new_build_capacity 
            storage_instance.new_build_e += new_build_capacity
    return None

@njit(fastmath=FASTMATH)    
def allocate_memory(storage_instance, intervals_count):
    if storage_instance.static_instance:
        raise_static_modification_error()
    storage_instance.dispatch_power = np.zeros(intervals_count, dtype=np.float64)
    storage_instance.stored_energy = np.zeros(intervals_count, dtype=np.float64)
    return None

@njit(fastmath=FASTMATH)
def initialise_stored_energy(storage_instance):
    if storage_instance.static_instance:
        raise_static_modification_error()
    storage_instance.stored_energy[-1] = 0.5*storage_instance.energy_capacity
    return None

@njit(fastmath=FASTMATH)    
def set_dispatch_max_t(storage_instance, interval: int, resolution: float, merit_order_idx: int):
    storage_instance.discharge_max_t = min(
        storage_instance.power_capacity, 
        storage_instance.stored_energy[interval-1] * storage_instance.discharge_efficiency / resolution
    )
    storage_instance.charge_max_t = min(
        storage_instance.power_capacity, 
        (storage_instance.energy_capacity - storage_instance.stored_energy[interval-1]) / storage_instance.charge_efficiency / resolution
    )

    if merit_order_idx == 0:
        storage_instance.node.discharge_max_t[0] = storage_instance.discharge_max_t
        storage_instance.node.charge_max_t[0] = storage_instance.charge_max_t
    else:
        storage_instance.node.discharge_max_t[merit_order_idx] = (
            storage_instance.node.discharge_max_t[merit_order_idx-1] + storage_instance.discharge_max_t
        )
        storage_instance.node.charge_max_t[merit_order_idx] = (
            storage_instance.node.charge_max_t[merit_order_idx-1] + storage_instance.charge_max_t
        )
    return None

@njit(fastmath=FASTMATH)    
def dispatch(storage_instance, interval: int, merit_order_idx: int) -> None:
    if merit_order_idx == 0:
        storage_instance.dispatch_power[interval] = (
            max(min(storage_instance.node.netload_t, storage_instance.discharge_max_t), 0.0) +
            min(max(storage_instance.node.netload_t, -storage_instance.charge_max_t), 0.0)
        )
    else:
        storage_instance.dispatch_power[interval] = (
            max(min(storage_instance.node.netload_t - storage_instance.node.discharge_max_t[merit_order_idx-1], storage_instance.discharge_max_t), 0.0) +
            min(max(storage_instance.node.netload_t + storage_instance.node.charge_max_t[merit_order_idx-1], -storage_instance.charge_max_t), 0.0)
        )
    storage_instance.node.storage_power[interval] += storage_instance.dispatch_power[interval]
    return None

@njit(fastmath=FASTMATH)    
def update_stored_energy(storage_instance, interval: int, resolution: float) -> None:
    storage_instance.stored_energy[interval] = storage_instance.stored_energy[interval-1] \
        - max(storage_instance.dispatch_power[interval], 0) / storage_instance.discharge_efficiency * resolution \
        - min(storage_instance.dispatch_power[interval], 0) * storage_instance.charge_efficiency * resolution 
    return None

@njit(fastmath=FASTMATH)    
def calculate_lt_discharge(storage_instance, interval_resolutions: NDArray[np.float64]) -> None:
    storage_instance.lt_discharge = sum(
        np.maximum(storage_instance.dispatch_power, 0)*interval_resolutions
    )

    storage_instance.line.lt_flows += sum(
        np.abs(storage_instance.dispatch_power)*interval_resolutions
    )
    return None

@njit(fastmath=FASTMATH)    
def calculate_lt_costs(storage_instance, years_float: float, year_count: int) -> float:
    ltcosts_m.calculate_annualised_build(storage_instance.lt_costs, storage_instance.energy_capacity, storage_instance.power_capacity, 0.0, storage_instance.cost, year_count, 'storage')
    ltcosts_m.calculate_fom(storage_instance.lt_costs, storage_instance.power_capacity, years_float, 0.0, storage_instance.cost, 'storage')
    ltcosts_m.calculate_vom(storage_instance.lt_costs, storage_instance.lt_discharge, storage_instance.cost)
    ltcosts_m.calculate_fuel(storage_instance.lt_costs, storage_instance.lt_discharge, 0, storage_instance.cost)
    return ltcosts_m.get_total(storage_instance.lt_costs)