import numpy as np
from numpy.typing import NDArray

from typing import Dict
from firm_ce.common.constants import JIT_ENABLED, FASTMATH
from firm_ce.system.topology import Node_InstanceType
from firm_ce.fast_methods import node_m
from firm_ce.common.jit_overload import njit

@njit(fastmath=FASTMATH)
def get_year_t_boundaries(static_instance, year: int) -> NDArray[np.int64]:
    if year < static_instance.year_count - 1:
        last_t = static_instance.year_first_t[year+1]
    else:
        last_t = static_instance.intervals_count
    return static_instance.year_first_t[year], last_t

@njit(fastmath=FASTMATH)
def set_year_energy_demand(static_instance, nodes_typed_dict: Node_InstanceType) -> None:
    for year in range(static_instance.year_count):
        first_t, last_t = get_year_t_boundaries(static_instance, year)
        for node in nodes_typed_dict.values():
            static_instance.year_energy_demand[year] += sum(node_m.get_data(node, "trace")[first_t:last_t]) * static_instance.resolution
    return None

@njit(fastmath=FASTMATH)
def unset_year_energy_demand(static_instance) -> None:
    static_instance.year_energy_demand = np.zeros(static_instance.year_count, dtype=np.float64)
    return None

@njit(fastmath=FASTMATH)
def check_reliability_constraint(static_instance, year: int, year_unserved_energy: float) -> bool:
    return (year_unserved_energy / static_instance.year_energy_demand[year]) <= static_instance.allowance