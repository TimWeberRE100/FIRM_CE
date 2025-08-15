import numpy as np
from numpy.typing import NDArray

from firm_ce.system.topology import Node
from firm_ce.common.constants import JIT_ENABLED, FASTMATH
from firm_ce.common.exceptions import (
    raise_static_modification_error,
    raise_getting_unloaded_data_error,
)
from firm_ce.fast_methods import generator_m
from firm_ce.common.typing import TypedDict, float64, string
from firm_ce.common.jit_overload import njit 

@njit(fastmath=FASTMATH)
def create_dynamic_copy(node_instance):
    node_copy = Node(
        False,
        node_instance.id,
        node_instance.order,
        node_instance.name
    )
    node_copy.data_status = node_instance.data_status 
    node_copy.data = node_instance.data # This remains static
    node_copy.residual_load = node_instance.residual_load.copy()
    return node_copy        

@njit(fastmath=FASTMATH)
def load_data(node_instance, 
                trace: NDArray[np.float64],):
    node_instance.data_status = "loaded"
    node_instance.data = trace
    node_instance.residual_load = trace.copy()
    return None

@njit(fastmath=FASTMATH)
def unload_data(node_instance):
    node_instance.data_status = "unloaded"
    node_instance.data = np.empty((0,), dtype=np.float64)
    return None

@njit(fastmath=FASTMATH)
def get_data(node_instance, data_type):
    if node_instance.data_status == "unloaded":
        raise_getting_unloaded_data_error()

    if data_type == "trace":
        return node_instance.data
    elif data_type == "residual_load":
        return node_instance.residual_load
    else:
        raise RuntimeError("Invalid data_type argument for Node.get_data(data_type).")
    return None

@njit(fastmath=FASTMATH)
def allocate_memory(node_instance):
    if node_instance.static_instance:
        raise_static_modification_error()
    node_instance.imports = np.zeros_like(node_instance.residual_load, dtype=np.float64)
    node_instance.exports = np.zeros_like(node_instance.residual_load, dtype=np.float64)
    node_instance.deficits = np.zeros_like(node_instance.residual_load, dtype=np.float64)
    node_instance.spillage = np.zeros_like(node_instance.residual_load, dtype=np.float64)

    node_instance.flexible_power = np.zeros_like(node_instance.residual_load, dtype=np.float64)
    node_instance.storage_power = np.zeros_like(node_instance.residual_load, dtype=np.float64)
    node_instance.flexible_energy = np.zeros_like(node_instance.residual_load, dtype=np.float64)
    node_instance.storage_energy = np.zeros_like(node_instance.residual_load, dtype=np.float64)
    return None

@njit(fastmath=FASTMATH)
def initialise_netload_t(node_instance, interval: int) -> None:
    node_instance.netload_t = get_data(node_instance, 'residual_load')[interval]
    return None

@njit(fastmath=FASTMATH)
def update_netload_t(node_instance, interval: int) -> None:
    # Note: exports are negative, so they add to load
    node_instance.netload_t = get_data(node_instance, 'residual_load')[interval] \
        - node_instance.imports[interval] - node_instance.exports[interval]
    return None

@njit(fastmath=FASTMATH)
def fill_required(node_instance) -> bool:
    return node_instance.fill > 1e-6

@njit(fastmath=FASTMATH)
def surplus_available(node_instance) -> bool:
    return node_instance.surplus > 1e-6

@njit(fastmath=FASTMATH)
def assign_storage_merit_order(node_instance, storages_typed_dict) -> None:
    storages_count = len(storages_typed_dict)
    temp_orders = np.full(storages_count, -1, dtype=np.int64)
    temp_durations = np.full(storages_count, -1, dtype=np.float64)

    idx = 0
    for storage_order, storage in storages_typed_dict.items():
        if storage.node.order == node_instance.order:
            temp_orders[idx] = storage_order
            temp_durations[idx] = storage.duration
            idx += 1

    if idx == 0:
        return

    temp_orders = temp_orders[:idx]
    temp_durations = temp_durations[:idx]

    sort_order = np.argsort(temp_durations)
    node_instance.storage_merit_order = temp_orders[sort_order]
    return None

@njit(fastmath=FASTMATH)
def assign_flexible_merit_order(node_instance, generators_typed_dict) -> None:
    generators_count = len(generators_typed_dict)
    temp_orders = np.full(generators_count, -1, dtype=np.int64)
    temp_marginal_costs = np.full(generators_count, -1, dtype=np.float64)

    idx = 0
    for generator_order, generator in generators_typed_dict.items():
        if not generator_m.check_unit_type(generator, 'flexible'):
            continue

        if generator.node.order == node_instance.order:
            temp_orders[idx] = generator_order
            temp_marginal_costs[idx] = generator.cost.vom + generator.cost.fuel_cost_mwh \
                + generator.cost.fuel_cost_h * 1000 * generator.unit_size
            idx += 1

    if idx == 0:
        return

    temp_orders = temp_orders[:idx]
    temp_marginal_costs = temp_marginal_costs[:idx]

    sort_order = np.argsort(temp_marginal_costs)
    node_instance.flexible_merit_order = temp_orders[sort_order]
    return None

@njit(fastmath=FASTMATH)
def check_remaining_netload(node_instance, interval: int, check_case: str) -> bool:
    if check_case == 'deficit':
        return node_instance.netload_t - node_instance.storage_power[interval] - node_instance.flexible_power[interval] > 1e-6
    elif check_case == 'spillage':
        return node_instance.netload_t - node_instance.storage_power[interval] - node_instance.flexible_power[interval] < 1e-6
    elif check_case == 'both':
        return abs(node_instance.netload_t - node_instance.storage_power[interval] - node_instance.flexible_power[interval]) > 1e-6
    return False