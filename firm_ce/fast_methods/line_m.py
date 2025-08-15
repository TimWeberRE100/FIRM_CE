import numpy as np

from firm_ce.system.topology import Line, Node
from firm_ce.common.constants import JIT_ENABLED, FASTMATH
from firm_ce.common.exceptions import (
    raise_static_modification_error,
)
from firm_ce.fast_methods import ltcosts_m

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
def create_dynamic_copy(line_instance, nodes_typed_dict, line_type):
    if line_type == "major":
        node_start_copy = nodes_typed_dict[line_instance.node_start.order]
        node_end_copy = nodes_typed_dict[line_instance.node_end.order]
    elif line_type == "major":
        node_start_copy = Node(False,-1,-1,"MINOR_NODE")
        node_end_copy = Node(False,-1,-1,"MINOR_NODE")
    
    line_copy = Line(
        False,
        line_instance.id,
        line_instance.order,
        line_instance.name,
        line_instance.length,
        node_start_copy,
        node_end_copy,
        line_instance.loss_factor,
        line_instance.max_build,
        line_instance.min_build,
        line_instance.capacity,
        line_instance.unit_type,
        line_instance.near_optimum_check,
        line_instance.group,
        line_instance.cost, # This remains static
    )
    line_copy.candidate_x_idx = line_instance.candidate_x_idx
    return line_copy

@njit(fastmath=FASTMATH)
def check_minor_line(line_instance) -> bool:
    return line_instance.id == -1

@njit(fastmath=FASTMATH)
def build_capacity(line_instance, new_build_power_capacity):
    if line_instance.static_instance:
        raise_static_modification_error()
    line_instance.capacity += new_build_power_capacity
    return None

@njit(fastmath=FASTMATH)
def allocate_memory(line_instance, intervals_count: int) -> None:
    if line_instance.static_instance:
        raise_static_modification_error()
    line_instance.flows = np.zeros(intervals_count, dtype=np.float64)
    return None

@njit(fastmath=FASTMATH)
def calculate_lt_flow(line_instance, resolution: float) -> None:
    line_instance.lt_flows = sum(np.abs(line_instance.flows)) * resolution
    return None

@njit(fastmath=FASTMATH)
def calculate_lt_costs(line_instance, years_float: float, year_count: int) -> float:
    ltcosts_m.calculate_annualised_build(line_instance.lt_costs, 0.0, line_instance.capacity, 0.0, line_instance.cost, year_count, 'line')
    ltcosts_m.calculate_fom(line_instance.lt_costs, line_instance.capacity, years_float, 0.0, line_instance.cost, 'line')
    ltcosts_m.calculate_vom(line_instance.lt_costs, line_instance.lt_flows, line_instance.cost)
    ltcosts_m.calculate_fuel(line_instance.lt_costs, line_instance.lt_flows, 0, line_instance.cost)
    return ltcosts_m.get_total(line_instance.lt_costs)

@njit(fastmath=FASTMATH)
def get_lt_losses(line_instance) -> float:
    return line_instance.lt_flows * line_instance.loss_factor * line_instance.length / 1000