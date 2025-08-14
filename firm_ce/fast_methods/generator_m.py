import numpy as np
from numpy.typing import NDArray
from typing import Union

from firm_ce.system.components import Generator
from firm_ce.common.constants import JIT_ENABLED, FASTMATH
from firm_ce.common.exceptions import (
    raise_static_modification_error,
    raise_getting_unloaded_data_error,
)
from firm_ce.fast_methods import node_m, ltcosts_m

if JIT_ENABLED:
    from numba import njit
else:
    def njit(func=None, **kwargs):
        if func is not None:
            return func
        def wrapper(f):
            return f
        return wrapper

@njit(fastmath=FASTMATH)
def create_dynamic_copy(generator_instance, nodes_typed_dict, lines_typed_dict):
    node_copy = nodes_typed_dict[generator_instance.node.order]
    line_copy = lines_typed_dict[generator_instance.line.order] 

    generator_copy = Generator(
        False,
        generator_instance.id, 
        generator_instance.order,
        generator_instance.name,
        generator_instance.unit_size,
        generator_instance.max_build,
        generator_instance.min_build,
        generator_instance.capacity,
        generator_instance.unit_type,
        generator_instance.near_optimum_check,
        node_copy,
        generator_instance.fuel, # This remains static
        line_copy,
        generator_instance.group,
        generator_instance.cost, # This remains static
    )
    generator_copy.data_status = generator_instance.data_status
    generator_copy.data = generator_instance.data # This remains static
    generator_copy.annual_constraints_data = generator_instance.annual_constraints_data # This remains static
    generator_copy.candidate_x_idx = generator_instance.candidate_x_idx
    generator_copy.lt_generation = generator_instance.lt_generation

    return generator_copy

@njit(fastmath=FASTMATH)
def build_capacity(generator_instance, new_build_power_capacity: float, resolution: float):
    if generator_instance.static_instance:
        raise_static_modification_error()     
    generator_instance.capacity += new_build_power_capacity   
    generator_instance.new_build += new_build_power_capacity 
    generator_instance.line.capacity += new_build_power_capacity 

    update_residual_load(generator_instance, new_build_power_capacity, resolution)     
    return None

@njit(fastmath=FASTMATH)    
def load_data(generator_instance, generation_trace: NDArray[np.float64], annual_constraints: NDArray[np.float64], resolution: float):
    generator_instance.data_status= "loaded"
    generator_instance.data = generation_trace
    generator_instance.annual_constraints_data = annual_constraints

    update_residual_load(generator_instance, generator_instance.initial_capacity, resolution)
    return None

@njit(fastmath=FASTMATH)    
def unload_data(generator_instance):
    generator_instance.data_status = "unloaded"
    generator_instance.data = np.empty((0,), dtype=np.float64)
    generator_instance.annual_constraints_data = np.empty((0,), dtype=np.float64)
    return None

@njit(fastmath=FASTMATH)    
def get_data(generator_instance, data_type: str) -> Union[NDArray[np.float64], None]:
    if generator_instance.data_status == "unloaded":
        raise_getting_unloaded_data_error()
        
    if data_type == "annual_constraints_data":
        return generator_instance.annual_constraints_data        
    elif data_type == "trace":
        return generator_instance.data        
    else:
        raise RuntimeError("Invalid data_type argument for Generator.get_data(data_type).")
    return None

@njit(fastmath=FASTMATH)
def allocate_memory(generator_instance, intervals_count):
    if generator_instance.static_instance:
        raise_static_modification_error()
    generator_instance.dispatch_power = np.zeros(intervals_count, dtype=np.float64)
    if len(get_data(generator_instance, 'annual_constraints_data')) > 0:
        generator_instance.remaining_energy = np.zeros(intervals_count, dtype=np.float64)
    return None

@njit(fastmath=FASTMATH)    
def update_residual_load(generator_instance, added_capacity: float, resolution: float) -> None:
    if get_data(generator_instance, "trace").shape[0] > 0 and added_capacity > 0.0:
        new_trace = get_data(generator_instance, "trace") * added_capacity
        node_m.get_data(generator_instance.node, "residual_load")[:] -= new_trace
        update_lt_generation(generator_instance, new_trace, resolution) 
    return None

@njit(fastmath=FASTMATH)    
def update_lt_generation(generator_instance, generation_trace: NDArray[np.float64], resolution: float) -> None:
    generator_instance.lt_generation += sum(generation_trace) * resolution
    generator_instance.line.lt_flows += generator_instance.lt_generation
    return None

@njit(fastmath=FASTMATH)    
def initialise_annual_limit(generator_instance, year, first_t) :
    if len(get_data(generator_instance, 'annual_constraints_data')) > 0:
        generator_instance.remaining_energy[first_t-1] = get_data(generator_instance, 'annual_constraints_data')[year]
    return None

@njit(fastmath=FASTMATH)    
def check_unit_type(generator_instance, unit_type: str) -> bool:
    return generator_instance.unit_type == unit_type

@njit(fastmath=FASTMATH)    
def set_flexible_max_t(generator_instance, interval: int, resolution: float, merit_order_idx: int) -> None:
    generator_instance.flexible_max_t = min(
        generator_instance.capacity, 
        generator_instance.remaining_energy[interval-1] / resolution
    )
    generator_instance.node.flexible_max_t[merit_order_idx] = generator_instance.node.flexible_max_t[merit_order_idx-1] + generator_instance.flexible_max_t
    return None

@njit(fastmath=FASTMATH)    
def dispatch(generator_instance, interval: int, merit_order_idx: int) -> bool:
    if merit_order_idx == 0:
        generator_instance.dispatch_power[interval] = min(
            max(generator_instance.node.netload_t - generator_instance.node.storage_power[interval], 0.0),
            generator_instance.flexible_max_t
        )
    else:
        generator_instance.dispatch_power[interval] = min(
            max(generator_instance.node.netload_t - generator_instance.node.storage_power[interval] - generator_instance.node.flexible_max_t[merit_order_idx-1], 0.0),
            generator_instance.flexible_max_t
        )
    generator_instance.node.flexible_power[interval] += generator_instance.dispatch_power[interval]
    return node_m.check_remaining_netload(generator_instance.node, interval, 'deficit')

@njit(fastmath=FASTMATH)    
def update_remaining_energy(generator_instance, interval: int, resolution: float) -> None:
    generator_instance.remaining_energy[interval] = generator_instance.remaining_energy[interval-1] - generator_instance.dispatch_power[interval] * resolution
    return None

@njit(fastmath=FASTMATH)    
def calculate_lt_generation(generator_instance, resolution: float) -> None:
    update_lt_generation(generator_instance, generator_instance.dispatch_power, resolution)
    generator_instance.unit_lt_hours = sum(np.ceil(generator_instance.dispatch_power/generator_instance.unit_size)) * resolution
    return None

@njit(fastmath=FASTMATH)    
def calculate_lt_costs(generator_instance, years_float: float) -> float:
    ltcosts_m.calculate_annualised_build(generator_instance.lt_costs, 0.0, generator_instance.capacity, 0.0, generator_instance.cost, 'generator')
    ltcosts_m.calculate_fom(generator_instance.lt_costs, generator_instance.capacity, years_float, 0.0, generator_instance.cost, 'generator')
    ltcosts_m.calculate_vom(generator_instance.lt_costs, generator_instance.lt_generation, generator_instance.cost)
    ltcosts_m.calculate_fuel(generator_instance.lt_costs, generator_instance.lt_generation, generator_instance.unit_lt_hours, generator_instance.cost)
    return ltcosts_m.get_total(generator_instance.lt_costs)