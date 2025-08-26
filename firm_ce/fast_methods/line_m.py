import numpy as np

from firm_ce.system.topology import Line, Node, Node_InstanceType, Line_InstanceType
from firm_ce.common.constants import FASTMATH
from firm_ce.common.exceptions import (
    raise_static_modification_error,
)
from firm_ce.fast_methods import ltcosts_m
from firm_ce.common.jit_overload import njit
from firm_ce.common.typing import DictType, int64, float64, unicode_type, boolean

@njit(fastmath=FASTMATH)
def create_dynamic_copy(line_instance: Line_InstanceType, 
                        nodes_typed_dict: DictType(int64, Node_InstanceType), 
                        line_type: unicode_type
                        ) -> Line_InstanceType:
    if line_type == "major":
        node_start_copy = nodes_typed_dict[line_instance.node_start.order]
        node_end_copy = nodes_typed_dict[line_instance.node_end.order]
    elif line_type == "minor":
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
def check_minor_line(line_instance: Line_InstanceType) -> boolean:
    return line_instance.id == -1

@njit(fastmath=FASTMATH)
def build_capacity(line_instance: Line_InstanceType, new_build_power_capacity: float64) -> None:
    if line_instance.static_instance:
        raise_static_modification_error()
    line_instance.capacity += new_build_power_capacity
    line_instance.new_build += new_build_power_capacity 
    return None

@njit(fastmath=FASTMATH)
def allocate_memory(line_instance: Line_InstanceType, intervals_count: int64) -> None:
    if line_instance.static_instance:
        raise_static_modification_error()
    line_instance.flows = np.zeros(intervals_count, dtype=np.float64)
    return None

@njit(fastmath=FASTMATH)
def calculate_lt_flow(line_instance: Line_InstanceType, interval_resolutions: float64[:]) -> None:
    line_instance.lt_flows = sum(np.abs(line_instance.flows)*interval_resolutions)
    return None

@njit(fastmath=FASTMATH)
def calculate_variable_costs(line_instance: Line_InstanceType) -> float64:
    ltcosts_m.calculate_vom(line_instance.lt_costs, line_instance.lt_flows, line_instance.cost)
    ltcosts_m.calculate_fuel(line_instance.lt_costs, line_instance.lt_flows, 0, line_instance.cost)
    return ltcosts_m.get_variable(line_instance.lt_costs)

@njit(fastmath=FASTMATH)
def calculate_fixed_costs(line_instance: Line_InstanceType, years_float: float64, year_count: int64) -> float64:
    ltcosts_m.calculate_annualised_build(line_instance.lt_costs, 0.0, line_instance.capacity, 0.0, line_instance.cost, year_count, 'line')
    ltcosts_m.calculate_fom(line_instance.lt_costs, line_instance.capacity, years_float, 0.0, line_instance.cost, 'line')
    return ltcosts_m.get_fixed(line_instance.lt_costs)

@njit(fastmath=FASTMATH)
def get_lt_losses(line_instance: Line_InstanceType) -> float64:
    return line_instance.lt_flows * line_instance.loss_factor * line_instance.length / 1000