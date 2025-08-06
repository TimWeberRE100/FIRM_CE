from typing import Dict, List, Tuple
import numpy as np
from numpy.typing import NDArray

from firm_ce.system.topology import (
    Node, Line, Network, Route,
    Node_InstanceType, Line_InstanceType, 
    Network_InstanceType, Route_InstanceType,
    )
from firm_ce.constructors.cost_cons import construct_UnitCost_object
from firm_ce.io.validate import is_nan
from firm_ce.common.constants import JIT_ENABLED

if JIT_ENABLED:
    from numba.typed.typeddict import Dict as TypedDict
    from numba.typed.typedlist import List as TypedList
    from numba.core.types import DictType, int64, UniTuple, ListType

def construct_Node_object(idx: int, order: int, node_name: str) -> Node_InstanceType:
    return Node(True, idx, order, node_name)

def construct_Line_object(line_dict: Dict[str, str], 
                          nodes_object_dict: TypedDict[int64,Node_InstanceType], 
                          order: int) -> Line_InstanceType:
    idx = int(line_dict['id'])
    name = str(line_dict['name'])
    length = int(line_dict['length']) 
    loss_factor = float(line_dict['loss_factor'])  
    max_build = float(line_dict['max_build']) 
    min_build = float(line_dict['min_build'])  
    capacity = float(line_dict['initial_capacity']) 
    unit_type = str(line_dict['unit_type'])
    near_opt = str(line_dict.get('near_optimum','')).lower() in ('true','1','yes')
    minor_node = construct_Node_object(-1,-1,"MINOR_NODE")

    raw_group = line_dict.get('range_group','')
    group = (
        name
        if raw_group is None
        or (isinstance(raw_group, float) and np.isnan(raw_group))
        or str(raw_group).strip() == ''
        else str(raw_group).strip()
    )

    node_start = next(
        (node for node in nodes_object_dict.values()
        if not is_nan(line_dict['node_start']) and node.name == str(line_dict['node_start'])),
        minor_node
    )

    node_end = next(
        (node for node in nodes_object_dict.values()
        if not is_nan(line_dict['node_end']) and node.name == str(line_dict['node_end'])),
        minor_node
    )
    
    cost = construct_UnitCost_object(
        float(line_dict['capex']),
        float(line_dict['fom']),
        float(line_dict['vom']),
        int(line_dict['lifetime']),
        float(line_dict['discount_rate']),
        transformer_capex=int(line_dict['transformer_capex']),
    )
    
    return Line(
        True,
        idx,
        order,
        name,
        length,
        node_start,
        node_end,
        loss_factor,
        max_build,
        min_build,
        capacity,
        unit_type,
        near_opt,
        group,
        cost,
    )

def construct_new_Route_object(
    initial_node: Node_InstanceType,
    new_node: Node_InstanceType,
    new_line: Line_InstanceType,
    line_direction: int,
    leg: int
):
    route_nodes = TypedList.empty_list(Node_InstanceType)
    route_lines = TypedList.empty_list(Line_InstanceType)
    
    route_nodes.append(new_node)
    route_lines.append(new_line)

    return Route(
        True,
        initial_node,
        route_nodes,
        route_lines,
        np.array([line_direction], dtype=np.int64),
        leg
    )

def extend_route(
    route: Route_InstanceType,
    new_node: Node_InstanceType,
    new_line: Line_InstanceType,
    line_direction: int,
    leg: int
) -> Route_InstanceType:
    route_nodes = route.nodes.copy()
    route_nodes.append(new_node)
    
    route_lines = route.lines.copy()
    route_lines.append(new_line)
    
    route_line_directions = list(route.line_directions)
    route_line_directions.append(line_direction)
    
    return Route(
        True,
        route.initial_node,
        route_nodes,
        route_lines,
        np.array(route_line_directions, dtype=np.int64),
        leg
    )

def get_routes_for_node(
        initial_node: Node_InstanceType,
        routes_typed_dict: DictType(UniTuple(int64,2), ListType(Route_InstanceType)),
        lines_object_dict: TypedDict[int64,Line_InstanceType],
        leg: int,
        ) -> ListType(Route_InstanceType):
    routes_to_node_curr_leg = TypedList.empty_list(Route_InstanceType)
    key = (initial_node.order, leg - 1)
    if key in routes_typed_dict:
        routes_to_node_prev_leg = routes_typed_dict[initial_node.order, leg-1].copy()
        for route in routes_to_node_prev_leg:        
            for line in lines_object_dict.values():
                if route.check_contains_line(line): # Remove loops
                    continue
                if line.node_start.order == route.nodes[-1].order:
                    if route.check_contains_node(line.node_end): # Remove loops
                        continue
                    new_route = extend_route(route, line.node_end, line, -1, leg)
                    routes_to_node_curr_leg.append(new_route)
                elif line.node_end.order == route.nodes[-1].order:
                    if route.check_contains_node(line.node_start): # Remove loops
                        continue
                    new_route = extend_route(route, line.node_start, line, 1, leg)
                    routes_to_node_curr_leg.append(new_route)
    else:              
        routes_to_node_prev_leg = TypedList.empty_list(Route_InstanceType)
        for line in lines_object_dict.values():
            if line.node_start.order == initial_node.order:
                new_route = construct_new_Route_object(initial_node, line.node_end, line, -1, leg)
                routes_to_node_curr_leg.append(new_route)
            elif line.node_end.order == initial_node.order:
                new_route = construct_new_Route_object(initial_node, line.node_start, line, 1, leg)
                routes_to_node_curr_leg.append(new_route)   
    return routes_to_node_curr_leg

def build_routes_typed_dict(
        networksteps_max: int, 
        nodes_object_dict: TypedDict[int64,Node_InstanceType],
        lines_object_dict: TypedDict[int64,Line_InstanceType],
        ) -> DictType(UniTuple(int64,2), ListType(Route_InstanceType)):
    
    routes_typed_dict = TypedDict.empty(
        key_type=UniTuple(int64, 2),
        value_type=ListType(Route_InstanceType)
    )

    for leg in range(networksteps_max):
        for node in nodes_object_dict.values():
            routes_typed_list = get_routes_for_node(
                node,
                routes_typed_dict,
                lines_object_dict,
                leg,
            )

            routes_typed_dict[(node.order, leg)] = routes_typed_list
    return routes_typed_dict

def construct_Network_object(
        nodes_imported_list: List[str],
        lines_imported_dict: Dict[str, Dict[str,str]],
        networksteps_max: int,
        ) -> Network_InstanceType:
    
    nodes = TypedDict.empty(
        key_type=int64,
        value_type=Node_InstanceType
    )
    for idx in range(len(nodes_imported_list)):
        nodes[idx] = construct_Node_object(idx, idx, nodes_imported_list[idx]) # Separate idx from order in fture version for consistency? 
    
    major_lines = TypedDict.empty(
        key_type=int64,
        value_type=Line_InstanceType
    )
    minor_lines = TypedDict.empty(
        key_type=int64,
        value_type=Line_InstanceType
    )
    order_major = 0
    order_minor = 0
    for idx in lines_imported_dict:
        if not is_nan(lines_imported_dict[idx]['node_start']) and not is_nan(lines_imported_dict[idx]['node_end']):
            major_lines[order_major] = construct_Line_object(lines_imported_dict[idx], nodes, order_major)
            order_major+=1
        else:
            minor_lines[order_minor] = construct_Line_object(lines_imported_dict[idx], nodes, order_minor)
            order_minor+=1
    
    routes = build_routes_typed_dict(networksteps_max, nodes, major_lines)
    
    return Network(
        True,
        nodes,
        major_lines,
        minor_lines,
        routes,
        networksteps_max
    )