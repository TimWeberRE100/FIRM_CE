from firm_ce.system.topology import Network, Node_InstanceType, Line_InstanceType, Route_InstanceType, routes_key_type, routes_list_type
from firm_ce.common.constants import JIT_ENABLED, FASTMATH
from firm_ce.common.exceptions import (
    raise_static_modification_error,
)
from firm_ce.fast_methods import line_m, node_m, route_m

if JIT_ENABLED:
    from numba import njit
    from numba.core.types import int64
    from numba.typed.typeddict import Dict as TypedDict
    from numba.typed.typedlist import List as TypedList
else:
    def njit(func=None, **kwargs):
        if func is not None:
            return func
        def wrapper(f):
            return f
        return wrapper

@njit(fastmath=FASTMATH)
def create_dynamic_copy(network_instance):
    nodes_copy = TypedDict.empty(
        key_type=int64,
        value_type=Node_InstanceType
    )
    major_lines_copy = TypedDict.empty(
        key_type=int64,
        value_type=Line_InstanceType
    )
    minor_lines_copy = TypedDict.empty(
        key_type=int64,
        value_type=Line_InstanceType
    )
    routes_copy = TypedDict.empty(
        key_type=routes_key_type,
        value_type=routes_list_type
    )

    for order, node in network_instance.nodes.items():
        nodes_copy[order] = node_m.create_dynamic_copy(node)

    for order, line in network_instance.major_lines.items():
        major_lines_copy[order] = line_m.create_dynamic_copy(line, nodes_copy, "major")

    for order, line in network_instance.minor_lines.items():
        minor_lines_copy[order] = line_m.create_dynamic_copy(line, nodes_copy, "minor")
    
    for tuple_key, routes_list in network_instance.routes.items():
        routes_copy[tuple_key] = TypedList.empty_list(Route_InstanceType)
        for route in routes_list:
            routes_copy[tuple_key].append(route_m.create_dynamic_copy(route, nodes_copy, major_lines_copy))

    network_copy = Network(
        False,
        nodes_copy,
        major_lines_copy,
        minor_lines_copy,
        routes_copy,
        network_instance.networksteps_max,
    )
    network_copy.major_line_count = network_instance.major_line_count
    return network_copy

@njit(fastmath=FASTMATH)
def build_capacity(network_instance, decision_x) -> None:
    if network_instance.static_instance:
        raise_static_modification_error()
    for line in network_instance.lines.values():
        line.capacity += decision_x[line.candidate_x_idx]
    return None

@njit(fastmath=FASTMATH)
def unload_data(network_instance):
    for node in network_instance.nodes.values():
        node_m.unload_data(node)
    return None

@njit(fastmath=FASTMATH)
def allocate_memory(network_instance, intervals_count: int) -> None:
    if network_instance.static_instance:
        raise_static_modification_error()
    for node in network_instance.nodes.values():
        node_m.allocate_memory(node)
    for line in network_instance.major_lines.values():
        line_m.allocate_memory(line, intervals_count)
    return None

@njit(fastmath=FASTMATH)
def check_remaining_netloads(network_instance, interval: int, check_case: str) -> bool:
    for node in network_instance.nodes.values():
        if node_m.check_remaining_netload(node, interval, check_case):
            return True
    return False

@njit(fastmath=FASTMATH)
def calculate_period_unserved_power(network_instance, first_t: int, last_t: int):
    unserved_power = 0
    for node in network_instance.nodes.values():
        unserved_power += sum(node.deficits[first_t:last_t+1])
    return unserved_power

@njit(fastmath=FASTMATH)
def reset_transmission(network_instance, interval: int) -> None:
    for line in network_instance.major_lines.values():
        line.flows[interval] = 0.0
    return None

@njit(fastmath=FASTMATH)
def reset_flow_updates(network_instance) -> None:
    for route_list in network_instance.routes.values():
        for route in route_list:
            route.flow_update = 0.0
    return None

@njit(fastmath=FASTMATH)
def check_route_surpluses(network_instance, fill_node: Node_InstanceType, leg: int) -> bool:
    # Check if final node in the route has a surplus available
    for route in network_instance.routes[fill_node.order, leg]:
        if node_m.surplus_available(route.nodes[-1]):
            return True
    return False

@njit(fastmath=FASTMATH)
def check_network_surplus(network_instance) -> bool:
    for node in network_instance.nodes.values():
        if node_m.surplus_available(node):
            return True
    return False

@njit(fastmath=FASTMATH)
def check_network_fill(network_instance) -> bool:
    for node in network_instance.nodes.values():
        if node_m.fill_required(node):
            return True
    return False

@njit(fastmath=FASTMATH)
def calculate_node_flow_updates(network_instance, fill_node: Node_InstanceType, leg: int, interval: int) -> None:
    fill_node.available_imports = 0.0
    for route in network_instance.routes[fill_node.order, leg]:
        route_m.calculate_flow_update(route, interval)
    return None

@njit(fastmath=FASTMATH)
def scale_flow_updates_to_fill(network_instance, fill_node: Node_InstanceType, leg: int) -> float:
    if fill_node.available_imports > fill_node.fill:
        scale_factor = fill_node.fill / fill_node.available_imports
        for route in network_instance.routes[fill_node.order, leg]:
            route.flow_update *= scale_factor
    return None

@njit(fastmath=FASTMATH)
def update_transmission_flows(network_instance, fill_node: Node_InstanceType, leg: int, interval: int) -> None:
    for route in network_instance.routes[fill_node.order, leg]:
        fill_node.imports[interval] += route.flow_update
        fill_node.fill -= route.flow_update
        route_m.update_exports(route, interval)
    return None

@njit(fastmath=FASTMATH)
def update_netloads(network_instance, interval: int) -> None:
    for node in network_instance.nodes.values():
        node_m.update_netload_t(node, interval)
    return None

@njit(fastmath=FASTMATH)
def reset_line_temp_flows(network_instance) -> None:
    for line in network_instance.major_lines.values():
        line.temp_leg_flows = 0.0
    return None

@njit(fastmath=FASTMATH)
def fill_with_transmitted_surpluses(network_instance, interval) -> None:        
    reset_flow_updates(network_instance) 
    if not (check_network_surplus(network_instance) and check_network_fill(network_instance)):
        return None
    
    for leg in range(network_instance.networksteps_max):
        for node in network_instance.nodes.values():
            if not node_m.fill_required(node):
                continue
            if len(network_instance.routes[node.order, leg]) == 0:
                continue
            if not check_route_surpluses(network_instance, node, leg):
                continue
            reset_line_temp_flows(network_instance)
            calculate_node_flow_updates(network_instance, node, leg, interval)
            scale_flow_updates_to_fill(network_instance, node, leg)
            update_transmission_flows(network_instance, node, leg, interval)       
    return None

@njit(fastmath=FASTMATH)
def set_node_fills_and_surpluses(network_instance, transmission_case: str, interval: int) -> None:
    if transmission_case == 'surplus':
        for node in network_instance.nodes.values():
            node.fill = max(node.netload_t, 0)
            node.surplus = -1*min(node.netload_t, 0)
    elif transmission_case == 'storage_discharge':
        for node in network_instance.nodes.values():
            node.fill = max(node.netload_t - node.storage_power[interval], 0)
            node.surplus = max(node.discharge_max_t[-1] - node.storage_power[interval], 0)
    elif transmission_case == 'flexible':
        for node in network_instance.nodes.values():
            node.fill = max(node.netload_t - node.storage_power[interval] - node.flexible_power[interval], 0)
            node.surplus = max(node.flexible_max_t[-1] - node.flexible_power[interval], 0)
    elif transmission_case == 'storage_charge':
        for node in network_instance.nodes.values():
            node.fill = max(node.charge_max_t[-1] + node.storage_power[interval], 0)
            node.surplus = -min(
                node.netload_t - min(node.storage_power[interval], 0),
                0
                )
    return None

@njit(fastmath=FASTMATH)
def calculate_spillage_and_deficit(network_instance, interval: int) -> None:
    for node in network_instance.nodes.values():
        node.deficits[interval] = max(node.netload_t - node.storage_power[interval] - node.flexible_power[interval], 0)
        node.spillage[interval] = min(node.netload_t - node.storage_power[interval] - node.flexible_power[interval], 0)
    return None

@njit(fastmath=FASTMATH)
def assign_storage_merit_orders(network_instance, storages_typed_dict) -> None: # storages_typed_dict: DictType(int64, Storage_InstanceType)
    for node in network_instance.nodes.values():
        node_m.assign_storage_merit_order(node, storages_typed_dict)
    return None

@njit(fastmath=FASTMATH)
def assign_flexible_merit_orders(network_instance, generators_typed_dict) -> None: # generators_typed_dict: DictType(int64, Generators_InstanceType)
    for node in network_instance.nodes.values():
        node_m.assign_flexible_merit_order(node, generators_typed_dict)
    return None

@njit(fastmath=FASTMATH)
def calculate_lt_flows(network_instance, resolution: float) -> None:
    for line in network_instance.major_lines.values():
        line_m.calculate_lt_flow(line, resolution)
    return None

@njit(fastmath=FASTMATH)
def calculate_lt_line_losses(network_instance) -> float:
    total_line_losses = 0.0
    for line in network_instance.major_lines.values():
        total_line_losses += line_m.get_lt_losses(line)
    for line in network_instance.minor_lines.values():
        total_line_losses += line_m.get_lt_losses(line)
    return total_line_losses