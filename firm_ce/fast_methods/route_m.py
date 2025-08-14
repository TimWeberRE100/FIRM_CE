from firm_ce.system.topology import Node_InstanceType, Line_InstanceType, Route
from firm_ce.common.constants import JIT_ENABLED, FASTMATH, NP_FLOAT_MAX

if JIT_ENABLED:
    from numba import njit
    from numba.core.types import DictType, int64
    from numba.typed.typedlist import List as TypedList
else:
    def njit(func=None, **kwargs):
        if func is not None:
            return func
        def wrapper(f):
            return f
        return wrapper

@njit(fastmath=FASTMATH)
def create_dynamic_copy(route_instance,
                        nodes_typed_dict: DictType(int64,Node_InstanceType),
                        lines_typed_dict: DictType(int64,Line_InstanceType)):
    nodes_list_copy = TypedList.empty_list(Node_InstanceType)
    lines_list_copy = TypedList.empty_list(Line_InstanceType)

    for node in route_instance.nodes:
        nodes_list_copy.append(nodes_typed_dict[node.order])

    for line in route_instance.lines:
        lines_list_copy.append(lines_typed_dict[line.order])
    
    return Route(False,
                nodes_typed_dict[route_instance.initial_node.order],
                nodes_list_copy,
                lines_list_copy,
                route_instance.line_directions.copy(),
                route_instance.legs)

@njit(fastmath=FASTMATH)
def check_contains_line(route_instance, new_line: Line_InstanceType) -> bool:
    for line in route_instance.lines:
        if new_line.order == line.order:
            return True
    return False

@njit(fastmath=FASTMATH)
def check_contains_node(route_instance, new_node: Node_InstanceType) -> bool:
    for node in route_instance.nodes:
        if new_node.order == node.order:
            return True
    return False

@njit(fastmath=FASTMATH)
def get_max_flow_update(route_instance, interval):
    max_flow = NP_FLOAT_MAX
    for leg in range(route_instance.legs+1):
        max_flow = min(
            max_flow, 
            route_instance.lines[leg].capacity - route_instance.line_directions[leg] * (route_instance.lines[leg].flows[interval] + route_instance.lines[leg].temp_leg_flows)
        )
    return max_flow

@njit(fastmath=FASTMATH)
def calculate_flow_update(route_instance, interval):
    route_instance.flow_update = min(
        route_instance.nodes[-1].surplus,
        get_max_flow_update(route_instance, interval)
    )
    route_instance.initial_node.available_imports += route_instance.flow_update

    # If multiple routes on the same leg use the same lines, they must be constrained by capacity committed for that leg
    for leg in range(route_instance.legs+1):
        route_instance.lines[leg].temp_leg_flows += route_instance.line_directions[leg] * route_instance.flow_update
    return None

@njit(fastmath=FASTMATH)
def update_exports(route_instance, interval: int) -> None:
    route_instance.nodes[-1].exports[interval] -= route_instance.flow_update
    route_instance.nodes[-1].surplus -= route_instance.flow_update
    
    for leg in range(route_instance.legs+1):
        route_instance.lines[leg].flows[interval] += route_instance.line_directions[leg] * route_instance.flow_update
    return None